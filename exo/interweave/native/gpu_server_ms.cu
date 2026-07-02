// gpu_server_ms.cu — MULTI-STREAM render-farm matmul server.
// Breaks the v6 serial wall: per-connection CUDA lane (own cublas handle + stream +
// scratch) so requests OVERLAP copy-in/GEMM/copy-out across lanes. Shared Q4_K weight
// cache is refcounted (eviction can't free a weight an in-flight GEMM is reading).
// PORT + GPU device configurable (argv1=port, env GPU_DEVICE) for the multi-GPU shard.
// FP16 tensor-op math gated on CC>=7.0 (M40 = CC5.2 has no tensor cores -> default math).
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <unordered_map>
#include <list>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#define MAGIC 0x47505534u
static int PORT = 8097;
static int G_DEV = 0;          // GPU device index (per-thread cudaSetDevice needed!)
static bool G_TENSOR = true;   // set from device compute capability in main()

__device__ __forceinline__ float f16d(unsigned short b){unsigned int s=(b>>15)&1u,e=(b>>10)&0x1Fu,m=b&0x3FFu,f;if(e==0u){if(m==0u)return 0.f;e=1u;while(!(m&0x400u)){m<<=1;e--;}m&=0x3FFu;f=(s<<31)|((e-15u+127u)<<23)|(m<<13);}else if(e==31u)return s?-1e30f:1e30f;else f=(s<<31)|((e-15u+127u)<<23)|(m<<13);float o;memcpy(&o,&f,4);return o;}
__device__ __forceinline__ void gsm(int j,const unsigned char*q,unsigned char*d,unsigned char*m){if(j<4){*d=q[j]&63u;*m=q[j+4]&63u;}else{*d=(q[j+4]&0xFu)|((q[j-4]>>6)<<4);*m=(q[j+4]>>4)|((q[j]>>6)<<4);}}
__global__ void dq(const unsigned char*data,__half*out,int nb){int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=nb)return;const unsigned char*blk=data+(size_t)b*144;__half*y=out+(size_t)b*256;float d=f16d(blk[0]|(blk[1]<<8)),dm=f16d(blk[2]|(blk[3]<<8));const unsigned char*sc=blk+4,*q=blk+16;int is=0,yi=0;for(int j=0;j<256;j+=64){unsigned char s1,m1,s2,m2;gsm(is,sc,&s1,&m1);gsm(is+1,sc,&s2,&m2);float d1=d*s1,n1=dm*m1,d2=d*s2,n2=dm*m2;for(int l=0;l<32;l++)y[yi+l]=__float2half(d1*(float)(q[l]&0xF)-n1);for(int l=0;l<32;l++)y[yi+32+l]=__float2half(d2*(float)(q[l]>>4)-n2);yi+=64;q+=32;is+=2;}}
__global__ void cvt(const float*in,__half*out,size_t n){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__float2half(in[i]);}

static int rall(int s,void*p,size_t n){char*b=(char*)p;size_t o=0;while(o<n){ssize_t r=recv(s,b+o,n-o,0);if(r<0){if(errno==EINTR)continue;return -1;}if(r==0)return -1;o+=r;}return 0;}
static int sall(int s,const void*p,size_t n){const char*b=(const char*)p;size_t o=0;while(o<n){ssize_t r=send(s,b+o,n-o,0);if(r<0){if(errno==EINTR)continue;return -1;}if(r==0)return -1;o+=r;}return 0;}

/* ---- shared Q4_K weight cache: REFCOUNTED so eviction never frees an in-flight weight ---- */
struct Ent{__half*d;size_t bytes;uint32_t M,K;int ref;std::list<uint32_t>::iterator it;};
static std::unordered_map<uint32_t,Ent> g_cache; static std::list<uint32_t> g_lru;
static size_t g_used=0,g_budget=0;
static pthread_mutex_t G_CACHE=PTHREAD_MUTEX_INITIALIZER;   // guards cache map / lru / used ONLY
static void evict_until(size_t need){
    // free from LRU back, but SKIP referenced (in-use) weights
    auto it=g_lru.end();
    while(g_used+need>g_budget && it!=g_lru.begin()){
        --it; auto c=g_cache.find(*it);
        if(c!=g_cache.end() && c->second.ref==0){
            cudaFree(c->second.d); g_used-=c->second.bytes;
            auto del=it; it=g_lru.erase(del); g_cache.erase(c);
        }
    }
}
static void egrow(void**p,size_t*cur,size_t need){if(need>*cur){if(*p)cudaFree(*p);cudaMalloc(p,need);*cur=need;}}

/* per-connection lane: its own GPU state -> concurrent execution */
struct Lane{ cublasHandle_t h; cudaStream_t st;
    unsigned char*dAq=0; __half*dAt=0,*dB=0,*dC=0; float*dBf=0;
    size_t kAq=0,kAt=0,kBf=0,kB=0,kC=0; };

void* handle(void*arg){
    int s=(int)(long)arg;int one=1;setsockopt(s,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
    cudaSetDevice(G_DEV);                         // CRITICAL: bind THIS thread to the right GPU
    Lane L;
    cublasCreate(&L.h); cudaStreamCreate(&L.st); cublasSetStream(L.h,L.st);
    cublasSetMathMode(L.h, G_TENSOR?CUBLAS_TENSOR_OP_MATH:CUBLAS_DEFAULT_MATH);
    void*hA=0,*hB=0,*hC=0;size_t cA=0,cB=0,cC=0;
    uint32_t H[8];
    while(rall(s,H,32)==0){
        if(H[0]!=MAGIC)break;
        uint32_t op=H[1],M=H[2],N=H[3],K=H[4],wid=H[7];
        size_t nb=((size_t)M*K+255)/256,nA=(size_t)M*K*2;
        if(op!=2){ size_t aB=nb*144; if(aB>cA){free(hA);hA=malloc(aB);cA=aB;} if(rall(s,hA,aB))break; }
        size_t bB=(size_t)K*N*4; if(bB>cB){free(hB);hB=malloc(bB);cB=bB;}
        if(rall(s,hB,bB))break;

        __half*Aw=0; bool fromCache=false;
        if(op==2){
            pthread_mutex_lock(&G_CACHE);
            auto it=g_cache.find(wid);
            if(it!=g_cache.end()){ Aw=it->second.d;M=it->second.M;K=it->second.K;
                it->second.ref++; fromCache=true;                 // PIN: in flight
                g_lru.erase(it->second.it);g_lru.push_front(wid);it->second.it=g_lru.begin(); }
            pthread_mutex_unlock(&G_CACHE);
        } else {
            // op!=2: dequantize provided A into this lane's scratch (or cache it for op==1)
            egrow((void**)&L.dAq,&L.kAq,nb*144);
            cudaMemcpyAsync(L.dAq,hA,nb*144,cudaMemcpyHostToDevice,L.st);
            if(op==1){
                __half*wd; pthread_mutex_lock(&G_CACHE);
                auto o=g_cache.find(wid);
                if(o!=g_cache.end()&&o->second.ref==0){cudaFree(o->second.d);g_used-=o->second.bytes;g_lru.erase(o->second.it);g_cache.erase(o);}
                evict_until(nA); cudaMalloc(&wd,nA); g_used+=nA;
                g_lru.push_front(wid); g_cache[wid]=Ent{wd,nA,M,K,1,g_lru.begin()}; // ref=1 in flight
                pthread_mutex_unlock(&G_CACHE);
                dq<<<(nb+127)/128,128,0,L.st>>>(L.dAq,wd,(int)nb); Aw=wd; fromCache=true;
            } else {
                egrow((void**)&L.dAt,&L.kAt,nA);
                dq<<<(nb+127)/128,128,0,L.st>>>(L.dAq,L.dAt,(int)nb); Aw=L.dAt;
            }
        }
        if(!Aw){ uint32_t R[4]={MAGIC,2,0,0}; if(sall(s,R,16))break; continue; }

        size_t cB_=(size_t)M*N*2;
        egrow((void**)&L.dBf,&L.kBf,bB); egrow((void**)&L.dB,&L.kB,(size_t)K*N*2); egrow((void**)&L.dC,&L.kC,cB_);
        cudaMemcpyAsync(L.dBf,hB,bB,cudaMemcpyHostToDevice,L.st);
        cvt<<<((size_t)K*N+255)/256,256,0,L.st>>>(L.dBf,L.dB,(size_t)K*N);
        float al=1,be=0;
        cublasStatus_t gst=cublasGemmEx(L.h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&al,L.dB,CUDA_R_16F,N,Aw,CUDA_R_16F,K,&be,L.dC,CUDA_R_16F,N,
                     CUBLAS_COMPUTE_32F, G_TENSOR?CUBLAS_GEMM_DEFAULT_TENSOR_OP:CUBLAS_GEMM_DEFAULT);
        if(gst!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"[ms] GEMM status=%d (M=%u N=%u K=%u)\n",(int)gst,M,N,K);fflush(stderr);}
        if(cB_>cC){free(hC);hC=malloc(cB_);cC=cB_;}
        cudaMemcpyAsync(hC,L.dC,cB_,cudaMemcpyDeviceToHost,L.st);
        cudaError_t ce=cudaStreamSynchronize(L.st);              // only THIS lane waits
        if(ce!=cudaSuccess){fprintf(stderr,"[ms] sync err=%s\n",cudaGetErrorString(ce));fflush(stderr);}

        if(fromCache){ pthread_mutex_lock(&G_CACHE); auto it=g_cache.find(wid); if(it!=g_cache.end()&&it->second.ref>0)it->second.ref--; pthread_mutex_unlock(&G_CACHE); }

        uint32_t R[4]={MAGIC,0,M,N}; if(sall(s,R,16)||sall(s,hC,cB_))break;
    }
    free(hA);free(hB);free(hC);
    if(L.dAq)cudaFree(L.dAq); if(L.dAt)cudaFree(L.dAt); if(L.dBf)cudaFree(L.dBf); if(L.dB)cudaFree(L.dB); if(L.dC)cudaFree(L.dC);
    cublasDestroy(L.h); cudaStreamDestroy(L.st); close(s); return 0;
}

int main(int argc,char**argv){
    if(argc>1)PORT=atoi(argv[1]);
    const char*ed=getenv("GPU_DEVICE"); int dev=ed?atoi(ed):0; G_DEV=dev; cudaSetDevice(dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,dev); G_TENSOR=(prop.major>=7);
    size_t fr=0,to=0; cudaMemGetInfo(&fr,&to);
    const char*e=getenv("GPU_CACHE_BUDGET_MB"); g_budget=e?(size_t)atoll(e)*1024*1024:(size_t)(fr*0.6);
    printf("[gpu-ms] :%d  dev=%d %s (CC%d.%d, tensor=%d)  per-lane streams  budget=%.0fMB (free=%.0fMB)\n",
           PORT,dev,prop.name,prop.major,prop.minor,G_TENSOR,g_budget/1e6,fr/1e6);fflush(stdout);
    int ls=socket(AF_INET,SOCK_STREAM,0);int one=1;setsockopt(ls,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
    sockaddr_in a;memset(&a,0,sizeof(a));a.sin_family=AF_INET;a.sin_port=htons(PORT);a.sin_addr.s_addr=INADDR_ANY;
    if(bind(ls,(sockaddr*)&a,sizeof(a))<0){perror("bind");return 1;}listen(ls,128);
    while(1){int c=accept(ls,0,0);if(c<0)continue;pthread_t t;pthread_create(&t,0,handle,(void*)(long)c);pthread_detach(t);}
}
