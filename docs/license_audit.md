# MatAnyone2 License Audit

Date: 2026-05-08

## Decision

MatAnyone2 is usable for a local, non-commercial OBS plugin prototype, but it is not commercially shippable without explicit written permission from the authors/contributors.

Do not distribute a commercial binary, embed the model weights in a commercial product, or use the plugin for income-generating streaming unless permission is obtained from the MatAnyone2 rightsholders.

## Source Reviewed

- MatAnyone2 GitHub repository license notice: https://github.com/pq-yang/MatAnyone2
- MatAnyone2 `LICENSE.txt`: https://raw.githubusercontent.com/pq-yang/MatAnyone2/main/LICENSE.txt
- PeiqingYang/MatAnyone2 Hugging Face model card: https://huggingface.co/PeiqingYang/MatAnyone2

## Findings

The upstream license is NTU S-Lab License 1.0. It permits redistribution and use in source and binary forms only for non-commercial purposes, provided the copyright notice, license conditions, and disclaimer are preserved.

Commercial use is not granted by the license. The license says to contact the contributors if redistribution or use for commercial purposes is required.

## Answers To Phase 0 Questions

- Streaming use, privately/non-commercially: permitted by the license text as non-commercial use.
- Distributing a compiled binary with model weights: permitted only for non-commercial distribution, and the binary distribution must reproduce the license notice, conditions, and disclaimer in documentation or other materials.
- Commercial use, including streaming for income: not permitted without explicit written permission.

## Gate Result

Proceed only if the intended use is local/non-commercial prototyping or non-commercial distribution with the required notices.

Pivot to RVM, RMBG, or another commercially permissive model if commercial streaming, paid distribution, sponsorship-driven streaming, or productization is required.
