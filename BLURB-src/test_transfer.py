from transformers import AutoModel, AutoTokenizer
from open_clip import create_model_from_pretrained, get_tokenizer
import torch


# 1. 加载原始PubMedBERT模型
base_model = AutoModel.from_pretrained(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
)

# 2. 加载BioMedCLIP的文本编码器权重（需要确认结构一致性）
clip = create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
)
text_model = clip[0].text

def transfer_weights(target_model, source_model):
    # 获取模型配置字典
    target_config = target_model.config.to_dict()
    source_config = source_model.config.to_dict()
    
    # 检查配置差异
    all_keys = set(target_config.keys()) | set(source_config.keys())
    has_diff = False
    
    print("模型配置差异比较:")
    print("=" * 50)
    
    for key in sorted(all_keys):
        if key not in target_config:
            print(f"目标模型缺少配置: {key} = {source_config[key]}")
            has_diff = True
        elif key not in source_config:
            print(f"源模型缺少配置: {key} = {target_config[key]}")
            has_diff = True
        elif target_config[key] != source_config[key]:
            print(f"配置不匹配 - {key}:")
            print(f"  目标模型: {target_config[key]}")
            print(f"  源模型: {source_config[key]}")
            has_diff = True
        elif target_config[key] == source_config[key]:
            print(f"  {key} 配置匹配")
    
    if has_diff:
        raise ValueError("模型结构不匹配")
    else:
        print("模型配置完全匹配！")
    
    # 逐层复制权重
    try:
        state_dict = source_model.state_dict()
        target_model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"权重迁移失败: {e}")
        # 可以添加更详细的权重层比较
        target_keys = set(target_model.state_dict().keys())
        source_keys = set(state_dict.keys())
        
        print("\n权重层差异:")
        print(f"目标模型特有层: {sorted(target_keys - source_keys)}")
        print(f"源模型特有层: {sorted(source_keys - target_keys)}")
        
        # 检查形状差异
        common_keys = target_keys.intersection(source_keys)
        shape_diff = False
        print("\n权重形状差异:")
        for key in sorted(common_keys):
            target_shape = target_model.state_dict()[key].shape
            source_shape = state_dict[key].shape
            if target_shape != source_shape:
                print(f"层 {key} 形状不匹配: 目标 {target_shape} vs 源 {source_shape}")
                shape_diff = True
        
        if not shape_diff:
            print("所有共同层的权重形状匹配")
        
        raise ValueError("权重迁移失败") from e
        
    return target_model

# 4. 执行权重迁移
enhanced_model = transfer_weights(base_model, text_model)

# 5. 验证迁移结果
test_text = "Histopathology shows malignant cells with irregular nuclei"
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
inputs = tokenizer(test_text, return_tensors="pt")

# 提取CLS向量比较相似度
with torch.no_grad():
    original_output = base_model(**inputs).last_hidden_state[:,0,:]
    enhanced_output = enhanced_model(**inputs).last_hidden_state[:,0,:]

similarity = torch.cosine_similarity(original_output, enhanced_output, dim=-1)
print(f"表示相似度: {similarity.item():.4f}")  # 预期接近1.0表示成功迁移