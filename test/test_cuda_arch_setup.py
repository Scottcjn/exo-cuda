import os
import unittest
from unittest.mock import patch

from cuda_arch_setup import generate_cudaarchs, get_arch_string, set_environment_variables


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
