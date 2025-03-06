# Union of Experts (UoE)

This repository hosts the code and models for the paper:

[Union of Experts: Adapting Hierarchical Routing to Equivalently Decomposed Transformer](https://www.arxiv.org/abs/2503.02495)

## Abstract
Mixture-of-Experts (MoE) enhances model performance while maintaining computational efficiency, making it well-suited for large-scale applications. However, expert in exist MoE paradigm works as an individual, thereby lacking high-quality expert interactions. Moreover, they have not been effectively extended to attention block, which constrains further efficiency improvements. To tackle these issues, we propose Union-of-Experts (UoE), which decomposes transformer into an equitant group of experts, and then implement selective routing on input data and experts. 

Our approach advances MoE design with four key innovations:(1) We conducted equitant expert decomposition on both MLP blocks and attention blocks based on matrix partition in tensor parallelism. (2) We developed two routing paradigms: patch-wise data selection and expert selection, to apply routing across different levels. (3) We design the architecture of UoE model, including Selective Multi-Head Attention (SMHA) and Union-of-MLP-Experts (UoME). (4) We develop parallel implementation of UoE’s routing and computation operation, and optimize efficiency based on the hardware processing analysis. 

Our experiments demonstrate that the UoE model surpass Full Attention, state-of-art MoEs and efficient transformers (including the model architecture of recently proposed DeepSeek-V3) in several tasks across image and natural language domains. In language modeling tasks, we achieve an average reduction of 2.38 in perplexity compared to the best-performed MoE method with an average of 76% FLOPs. In Long Range Arena benchmark, we recorded an average score that is at least 0.68% higher than all comparison models including Full Attention, MoEs, and transformer variants, with only 50% FLOPs of the best MoE method. In image classification, our model yielded an average accuracy improvement of 1.75% than the best model while maintaining comparable FLOPs.

# Updates
- March 5, 2025: Release the code for [UoE model](./language_modeling/model) and [language modeling experiments](./language_modeling)

More introduction to the code will be further improved later. We will also open source the remaining experimental parts of the paper. If you would like to learn more about our project, please wait. We will complete the construction of the code repository soon. 

Thank you!
