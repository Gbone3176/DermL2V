import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from open_clip import create_model_from_pretrained

class BiomedCLIPForSequenceClassification(PreTrainedModel):
    """
    BiomedCLIP text encoder model adapted for sequence classification tasks.
    This model adds a classification head on top of the text encoder from BiomedCLIP.
    """
    
    def __init__(self, config):
        """
        Initialize the model.
        
        Args:
            config: Model configuration object with additional attributes:
                   - num_labels: Number of classes for classification
        """
        super().__init__(config)
        
        # Load the BiomedCLIP model
        clip_model, _ = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        # Get the text encoder part
        self.text_model = clip_model.text
        
        # Freeze the parameters of the text model if needed
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # Get the dimension of text embeddings
        self.hidden_size = self.text_model.config.hidden_size
        
        # Create classification head
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(self.hidden_size, config.num_labels)
        
        # Initialize weights of the classifier
        self._init_weights(self.classifier)
        
    def _init_weights(self, module):
        """Initialize the weights of the classifier"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for computing the sequence classification loss
            
        Returns:
            SequenceClassifierOutput with loss, logits, and optionally hidden states and attentions
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get text features from the text encoder
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get the [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Apply classification head
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.view(-1), labels.view(-1))
            else:
                if labels.shape == logits.shape:
                    # Multi-label classification
                    loss_fn = nn.BCEWithLogitsLoss()
                    loss = loss_fn(logits, labels.float())
                else:
                    # Single-label classification
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, *model_args, **kwargs):
        """
        Create an instance of the model from a pretrained BiomedCLIP model.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model or model identifier
            
        Returns:
            An instance of BiomedCLIPForSequenceClassification
        """
        config = kwargs.pop("config", None)
        if config is None:
            # Create a minimal config for the classification model
            from transformers import PretrainedConfig
            num_labels = kwargs.pop("num_labels", 2)
            config = PretrainedConfig(num_labels=num_labels)
        
        model = cls(config)
        return model