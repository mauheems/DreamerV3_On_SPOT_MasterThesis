# Licenses & Attribution

This repository contains code from **three separate works**, stacked on top of each other.
All three are MIT licensed. The copyright holders and their contributions are listed below.

---

## 1. DreamerV3 — Copyright (c) 2023 Danijar Hafner

The base world model framework. The original codebase is at:
https://github.com/danijar/dreamerv3

MIT License applies. The full license text is preserved in:
`dreamer_SPOT_implementation/informed-dreamer/LICENSE`

---

## 2. Informed Dreamer — Lambrechts, Bolland & Ernst (2024)

Extends DreamerV3 for Informed POMDPs. Built directly on top of the DreamerV3 codebase
(with modifications). Original repository:
https://github.com/gaspardlambrechts/informed-dreamer

If citing, please use:
```bibtex
@article{lambrechts2024informed,
    title={Informed {POMDP}: {L}everaging Additional Information in Model-Based {RL}},
    author={Lambrechts, Gaspard and Bolland, Adrien and Ernst, Damien},
    journal={Reinforcement Learning Journal},
    volume={1}, issue={1}, year={2024}
}
```

---

## 3. SPOT Robot Adaptations — Copyright (c) 2026 Maurits Heemskerk

Extends Informed Dreamer with Boston Dynamics SPOT robot integration.
All new files and modifications are listed in `dreamer_SPOT_implementation/README.md`.

Key additions:
- `dreamerv3/embodied/envs/spot_live.py` — live deployment interface
- `dreamerv3/embodied/run/train_offline.py` — offline training loop
- `dreamerv3/train_online.py` — online training entrypoint
- `dreamerv3/embodied/envs/spot.py` — SPOT environment wrapper (modified)
- `dreamerv3/agent.py` — reward & observation integration (modified)
- `dreamerv3/jaxagent.py` — deployment inference (modified)
- `dreamerv3/configs.yaml` — SPOT training configs (modified)
- `validate_episodes.py` — data validation tool
- All notebooks, scripts, and configs in `dreamer_SPOT_implementation/`

---

## MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notices and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
