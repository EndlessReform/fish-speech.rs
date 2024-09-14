---
tags:
- text-to-speech
license: cc-by-nc-sa-4.0
language:
- zh
- en
- de
- ja
- fr
- es
- ko
- ar
pipeline_tag: text-to-speech
inference: false
extra_gated_prompt: >-
  You agree to not use the model to generate contents that violate DMCA or local
  laws.
extra_gated_fields:
  Country: country
  Specific date: date_picker
  I agree to use this model for non-commercial use ONLY: checkbox
---


# Fish Speech V1.4

**Fish Speech V1.4** is a leading text-to-speech (TTS) model trained on 700k hours of audio data in multiple languages.

Supported languages:
- English (en) ~300k hours 
- Chinese (zh) ~300k hours
- German (de) ~20k hours
- Japanese (ja) ~20k hours
- French (fr) ~20k hours
- Spanish (es) ~20k hours
- Korean (ko) ~20k hours
- Arabic (ar) ~20k hours

Please refer to [Fish Speech Github](https://github.com/fishaudio/fish-speech) for more info.  
Demo available at [Fish Audio](https://fish.audio/).

## Citation

If you found this repository useful, please consider citing this work:

```
@misc{fish-speech-v1.4,
  author = {Shijia Liao, Tianyu Li, etc},
  title = {Fish Speech V1.4},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fishaudio/fish-speech}}
}
```

## License

This model is permissively licensed under the BY-CC-NC-SA-4.0 license.
The source code is released under BSD-3-Clause license.
