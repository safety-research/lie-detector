<div align="center">
<img src="assets/fig1.png" alt="mask_figure_1" width="90%">

# The MASK Benchmark: Disentangling Honesty from Accuracy in AI Systems

🌐 [Website](https://www.mask-benchmark.ai) | 📄 [Paper](https://mask-benchmark.ai/paper) | 🤗 [Dataset](https://huggingface.co/datasets/cais/mask)

<img src="./assets/cais_logo.svg" width="30" />&nbsp;&nbsp;&nbsp;&&nbsp;&nbsp;&nbsp;<img src="./assets/scale.svg" width="100"/>

</div>

This repository contains the implementation for MASK (Model Alignment between Statements and Knowledge), a benchmark designed to evaluate honesty in large language models by testing whether they contradict their own beliefs when pressured to lie. MASK disentangles honesty from factual accuracy, using a comprehensive evaluation pipeline to measure how consistently models respond when incentivized to provide false information across various scenarios. We find that scaling pre-training does not improve model honesty.

## Dataset

The MASK Dataset is available for download on Hugging Face at [🤗 cais/mask](https://huggingface.co/datasets/cais/mask).

## Evaluation Framework

For details about the evaluation framework, please see the [MASK Evaluation README](mask/README.md).

## Citation

If you find this useful in your research, please consider citing:

```bibtex
@misc{ren2025maskbenchmarkdisentanglinghonesty,
  title={The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems}, 
  author={Richard Ren and Arunim Agarwal and Mantas Mazeika and Cristina Menghini and Robert Vacareanu and Brad Kenstler and Mick Yang and Isabelle Barrass and Alice Gatti and Xuwang Yin and Eduardo Trevino and Matias Geralnik and Adam Khoja and Dean Lee and Summer Yue and Dan Hendrycks},
  year={2025},
  eprint={2503.03750},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2503.03750}, 
}
```