import torch.nn as nn
import torch
from typing import Optional, Union, Tuple
from transformers import ChineseCLIPVisionModel, ChineseCLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling


class ChineseCLIPVisionMultiFrameModel(ChineseCLIPVisionModel):
    config_class = ChineseCLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: ChineseCLIPVisionConfig):
        super().__init__(config)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_frame: int = 3,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, ChineseCLIPVisionModel

        >>> model = ChineseCLIPVisionModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        >>> processor = CLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

        >>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        B, C, H, W = pixel_values.shape
        pixel_values = pixel_values.view(B * num_frame, C // num_frame, H, W)

        model_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            pooled_output = model_outputs[1].view(B, num_frame, -1).mean(dim=1)
            return (model_outputs[0], pooled_output) + pooled_output[2:]

        return BaseModelOutputWithPooling(
            last_hidden_state=model_outputs.last_hidden_state,
            pooler_output=model_outputs.pooler_output.view(B, num_frame, -1).mean(dim=1),
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )