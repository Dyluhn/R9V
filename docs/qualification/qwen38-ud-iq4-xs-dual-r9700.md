# Qwen3.8 UD-IQ4_XS dual-R9700 development qualification

The release candidate exposes OpenAI-compatible text, tools, and one-image
inputs with MTP2 and 128K context. During development, the reference machine
sustained:

| Context | PP | TG |
|---:|---:|---:|
| 8,202 + 256 | 90.77 | 78.38 |
| 32,778 + 256 | 105.44 | 79.27 |
| 65,546 + 256 | 133.95 | 78.57 |
| 120,010 + 234 | 153.24 | 77.82 |

The 120K run ended at 234 completion tokens. Rank 0 was the display card;
rank 1 used the LRU16 expert cache. PLE used SSD residency. R4D was disabled.

These measurements qualify the local exact profile and reference topology.
They are not the final release benchmark: public package upload followed by a
clean recursive-clone benchmark is still required.
