import os
import unittest
from unittest.mock import patch, MagicMock

from cuda_arch_setup import (
    detect_cuda_gpus,
    generate_cudaarchs,
    get_arch_string,
    set_environment_variables,
)


class TestDetectCudaGpus(unittest.TestCase):
  def _run_with_nvidia_smi(self, csv_line):
    proc = MagicMock(returncode=0, stdout=csv_line + "\n")
    with patch("cuda_arch_setup.subprocess.run", return_value=proc):
      return detect_cuda_gpus()

  def test_compute_cap_fraction_is_rounded_not_truncated(self):
    # Float(8.6) - 8 == 0.5999999... ; int() would floor to minor=5 (sm_85,
    # a non-existent arch). Real RTX 3090 is sm_86.
    gpu = self._run_with_nvidia_smi("NVIDIA GeForce RTX 3090, 8.6, 550.00, 24576")[0]
    self.assertEqual((gpu["major"], gpu["minor"]), (8, 6))
    self.assertEqual(get_arch_string((gpu["major"], gpu["minor"])), "sm_86")

  def test_pascal_and_jetson_caps_round_correctly(self):
    g1080 = self._run_with_nvidia_smi("NVIDIA GeForce GTX 1080, 6.1, 550.00, 8192")[0]
    self.assertEqual((g1080["major"], g1080["minor"]), (6, 1))
    orin = self._run_with_nvidia_smi("Jetson AGX Orin, 8.7, 540.00, 32768")[0]
    self.assertEqual((orin["major"], orin["minor"]), (8, 7))


class TestCudaArchSetup(unittest.TestCase):
  def test_get_arch_string_uses_known_and_fallback_architectures(self):
    self.assertEqual(get_arch_string((7, 5)), "sm_75")
    self.assertEqual(get_arch_string((8, 9)), "sm_89")
    self.assertEqual(get_arch_string((10, 2)), "sm_102")

  def test_generate_cudaarchs_uses_defaults_when_no_gpus_are_detected(self):
    self.assertEqual(generate_cudaarchs([]), "70;75;80;86")

  def test_generate_cudaarchs_deduplicates_and_sorts_known_gpu_architectures(self):
    gpus = [
      {"major": 8, "minor": 6},
      {"major": 7, "minor": 5},
      {"major": 8, "minor": 6},
      {"major": 5, "minor": 0},
    ]

    self.assertEqual(generate_cudaarchs(gpus), "50;75;86")

  def test_generate_cudaarchs_falls_back_when_only_unknown_gpus_are_detected(self):
    gpus = [
      {"major": 9, "minor": 9},
      {"major": 10, "minor": 1},
    ]

    self.assertEqual(generate_cudaarchs(gpus), "70;75;80;86")

  def test_set_environment_variables_updates_cuda_build_flags(self):
    gpus = [{"major": 7, "minor": 5}]

    with patch.dict(os.environ, {}, clear=True):
      set_environment_variables(gpus)

      self.assertEqual(os.environ["CUDAARCHS"], "75")
      self.assertEqual(os.environ["TORCH_CUDA_ARCH_LIST"], "75")


if __name__ == "__main__":
  unittest.main()
