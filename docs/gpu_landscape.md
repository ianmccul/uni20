# GPU Landscape for Tensor Networks (2026-04)

This note is a hardware-buying snapshot for Uni20 developers who care about tensor-network workloads.

It is not a statement of current Uni20 backend support. As of April 8, 2026, Uni20's CPU and BLAS paths are the usable production paths, while CUDA and broader heterogeneous execution remain partial.

Refresh this note before spending real money. The workstation GPU market changes quickly.

## What matters most for tensor networks

For tensor-network work, the most important GPU properties are usually:

- Memory capacity.
  Large bond dimensions and large intermediate tensors hit memory limits before they hit peak FLOPS.
- Memory bandwidth.
  Many contraction-heavy workloads are limited by moving data through memory rather than by pure arithmetic throughput.
- FP64 throughput.
  This matters most for numerically sensitive physics workloads, large condition-number problems, and codes that rely on double precision throughout.
- Interconnect.
  Multi-GPU tensor-network work benefits from fast peer-to-peer links. PCIe-only scaling is often much worse than NVLink or server-class fabric.
- Software stack.
  For tensor networks specifically, vendor libraries matter. NVIDIA has an official tensor-network stack. AMD has lower-level tensor primitives, but a weaker official tensor-network story.

## Software ecosystem

The current software landscape still strongly favors NVIDIA for tensor networks:

- NVIDIA ships [cuTensorNet](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html) as part of the [cuQuantum SDK](https://developer.nvidia.com/cuquantum-sdk). It explicitly targets tensor-network contraction, path optimization, slicing, MPS workflows, decompositions, and distributed execution.
- NVIDIA's official docs list support for `Turing`, `Ampere`, `Ada`, `Hopper`, and `Blackwell` GPUs.
- AMD ships [hipTensor](https://rocm.docs.amd.com/projects/hipTensor/en/docs-7.1.0/conceptual/programmers-guide.html), which provides tensor primitives such as contraction, permutation, and reduction. This is useful, but it is not the same as an official tensor-network library comparable to cuTensorNet.

Practical consequence:

- If you want the strongest out-of-the-box vendor support for tensor-network development, NVIDIA is the safer choice.
- If you want ROCm, treat the software stack as lower-level and less specialized for tensor-network workflows.

## Current GPU categories

### 1. Current workstation GPUs

These are the easiest cards to deploy in a desk-side workstation: active cooling, display outputs, vendor workstation support, and straightforward OEM integration.

| GPU | Memory | Precision story | Why it is interesting | Main caveats |
|---|---:|---|---|---|
| [NVIDIA RTX PRO 6000 Blackwell Workstation Edition](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/workstation-datasheet-blackwell-rtx-pro6000-x-nvidia-us-3519208-web.pdf) | 96 GB GDDR7 ECC | Official datasheet emphasizes `125 TFLOPS` FP32 and AI throughput; FP64 is not a headline spec | Huge local memory, strong workstation packaging, current flagship single-GPU desk-side option | `600 W`, expensive, and not positioned as an FP64-first part |
| [NVIDIA RTX 6000 Ada Generation](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/proviz-print-rtx6000-datasheet-web-2504660.pdf) | 48 GB GDDR6 ECC | Official datasheet emphasizes `91.1 TFLOPS` FP32; FP64 is not a headline spec | Mature workstation deployment, active cooling, strong CUDA ecosystem | Much better for AI / rendering / graphics than for FP64-heavy tensor-network work |
| [AMD Radeon PRO W7900](https://www.amd.com/content/dam/amd/en/documents/partner-hub/radeon-pro/amd-radeon-pro-w7900-dual-slot-datasheet-competitive.pdf) | 48 GB GDDR6 ECC | Official datasheet lists `61.32 TFLOPS` FP32 and `1.92 TFLOPS` FP64 | Active-cooled workstation card with large memory and Linux support | Official ROCm / tensor-network ecosystem is weaker; FP64 is far below old GV100-class cards and data-center accelerators |

Takeaway:

- The workstation market now optimizes for AI inference, rendering, visualization, and large local memory.
- It no longer optimizes for strong FP64 in the way older CAE / scientific workstation cards sometimes did.

### 2. Legacy workstation outlier

There is still one older card worth remembering because it sits in a niche that newer workstation cards mostly abandoned:

| GPU | Memory | FP64 | Why it still matters |
|---|---:|---:|---|
| [NVIDIA Quadro GV100](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/productspage/quadro/quadro-desktop/quadro-volta-gv100-data-sheet-us-nvidia-704619-r3-web.pdf) | 32 GB HBM2 ECC | `7.4 TFLOPS` | Active cooling, display outputs, workstation packaging, NVLink support for 2 GPUs, and much stronger FP64 than modern workstation graphics cards |

For a developer who already owns GV100s, the bad news is age and smaller memory by modern standards.
The good news is that there is still no clean modern workstation replacement for "active-cooled desk-side card with genuinely strong FP64."

### 3. Data-center PCIe accelerators

If FP64 is the priority, this is where the market moved.

| GPU | Memory | FP64 | Why it is interesting | Main caveats |
|---|---:|---:|---|---|
| [NVIDIA A100 80GB PCIe](https://www.nvidia.com/en-us/data-center/a100/) | 80 GB HBM2e | `9.7 TFLOPS` FP64, `19.5 TFLOPS` FP64 Tensor Core | Much stronger HPC story than workstation GPUs, HBM bandwidth, optional 2-GPU NVLink bridge | Data-center part, no display outputs, usually awkward in a normal workstation buying path |
| [AMD Instinct MI210](https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/product-briefs/instinct-mi210-brochure.pdf) | 64 GB HBM2e | `22.6 TFLOPS` FP64 | Very strong FP64 on a PCIe card, large memory, clearly HPC-oriented | Data-center accelerator, not a normal workstation graphics card, ROCm software tradeoffs |

Inference from these product positions:

- Strong FP64 is still available.
- It is just mostly sold as a server or data-center accelerator, not as a workstation graphics card.

## What this means in practice

### If you want a current desk-side workstation

The realistic choices are current workstation graphics cards such as:

- `RTX 6000 Ada`
- `RTX PRO 6000 Blackwell`
- `Radeon PRO W7900`

These are easiest to cool, easiest to power, and easiest to buy inside OEM workstation systems.
For tensor networks, they are mainly attractive because of memory size and deployment convenience, not because of elite FP64.

### If you need strong FP64

There is no obvious modern desk-side workstation successor to the Quadro GV100.

Your options are usually:

- keep older cards like `GV100` longer than you otherwise would
- move to data-center PCIe accelerators such as `A100 PCIe` or `MI210`
- push the most FP64-sensitive jobs back to CPU or server-class systems

### If you care most about vendor tensor-network software

NVIDIA remains the easier path because:

- `cuTensorNet` is real, current, and specifically about tensor networks
- `cuQuantum` is actively maintained
- workstation, server, and cluster deployment all sit on the same CUDA software family

AMD has improved low-level tensor primitives, but the official stack is still less compelling for tensor-network users specifically.

## A short buying guide

If you are choosing hardware in 2026 for tensor-network work:

- Choose a current NVIDIA workstation GPU if you want the smoothest local development environment and the strongest vendor tensor-network software stack.
- Choose a current AMD workstation GPU only if ROCm is a deliberate requirement and you are comfortable with a less specialized tensor-network ecosystem.
- Keep older GV100-class cards if FP64 matters more than age, display standards, or AI-oriented features.
- Move to A100 / MI210 / server-class accelerators if FP64 and HBM matter more than workstation convenience.

## Sources

- [NVIDIA cuTensorNet documentation](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html)
- [NVIDIA cuQuantum SDK overview](https://developer.nvidia.com/cuquantum-sdk)
- [AMD hipTensor documentation](https://rocm.docs.amd.com/projects/hipTensor/en/docs-7.1.0/conceptual/programmers-guide.html)
- [NVIDIA RTX PRO 6000 Blackwell Workstation Edition datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/workstation-datasheet-blackwell-rtx-pro6000-x-nvidia-us-3519208-web.pdf)
- [NVIDIA RTX 6000 Ada Generation datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/proviz-print-rtx6000-datasheet-web-2504660.pdf)
- [NVIDIA Quadro GV100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/productspage/quadro/quadro-desktop/quadro-volta-gv100-data-sheet-us-nvidia-704619-r3-web.pdf)
- [NVIDIA A100 product page](https://www.nvidia.com/en-us/data-center/a100/)
- [AMD Radeon PRO W7900 dual-slot datasheet](https://www.amd.com/content/dam/amd/en/documents/partner-hub/radeon-pro/amd-radeon-pro-w7900-dual-slot-datasheet-competitive.pdf)
- [AMD Instinct MI210 product brief](https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/product-briefs/instinct-mi210-brochure.pdf)
