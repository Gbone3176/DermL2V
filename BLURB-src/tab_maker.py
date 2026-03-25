import os
import json
import re
from datetime import datetime
from argparse import ArgumentParser

def parse_metrics_file(file_path):
    """解析指标文件并提取结构化数据"""
    metrics = {}
    current_task = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 识别任务类型行
            task_match = re.match(r"^(\w+)\sTask Metrics:", line)
            if task_match:
                current_task = task_match.group(1)
                continue
                
            # 解析数据集指标行
            dataset_match = re.match(r"^(\S+?)_hf:\s*({.*?})\s*$", line)
            if dataset_match and current_task:
                dataset, data_str = dataset_match.groups()
                try:
                    # 转换单引号为双引号并处理特殊字符
                    data_str = data_str.replace("'", "\"")
                    data = json.loads(data_str)
                    metrics[dataset] = {k.lower(): v for k, v in data.items()}
                except json.JSONDecodeError:
                    continue
    return metrics

def calculate_blurb_score(metrics_dict):
    """计算BLURB基准分数"""
    task_config = {
        "NER": {
            "datasets": ["BC2GM_hf", "BC5CDR-chem_hf", "BC5CDR-disease_hf",
                        "JNLPBA_hf", "NCBI-disease_hf"],
            "metric_keys": ["test_f1"] * 5
        },
        "PICO": {
            "datasets": ["ebmnlp_hf"],
            "metric_keys": ["test_macro_f1"]
        },
        "RE": {
            "datasets": ["chemprot_hf", "DDI_hf", "GAD_hf"],
            "metric_keys": ["test_f1"] * 3
        },
        "SS": {
            "datasets": ["BIOSSES_hf"],
            "metric_keys": ["test_pearsonr"]
        },
        "DC": {
            "datasets": ["HoC_hf"],
            "metric_keys": ["test_f1"]
        },
        "QA": {
            "datasets": ["bioasq_hf", "pubmedqa_hf"],
            "metric_keys": ["test_accuracy"] * 2
        }
    }

    task_scores = {}
    
    for task, config in task_config.items():
        scores = []
        for dataset, metric_key in zip(config["datasets"], config["metric_keys"]):
            if dataset in metrics_dict:
                metric_value = metrics_dict[dataset].get(metric_key.lower())
                if metric_value is not None:
                    scores.append(float(metric_value))
        
        if scores:
            task_scores[task] = sum(scores) / len(scores)
        else:
            task_scores[task] = 0.0

    blurb_score = sum(task_scores.values()) / len(task_scores)
    
    return {
        "task_scores": task_scores,
        "blurb_score": round(blurb_score * 100, 2)
    }

def generate_markdown_report(results, output_file):
    """生成Markdown格式的报告"""
    task_order = ["NER", "PICO", "RE", "SS", "DC", "QA"]
    
    # 构建表头
    header = "| Model | " + " | ".join(task_order) + " | BLURB Score |\n"
    separator = "|-------|" + "|".join([":-------:"] * len(task_order)) + "|-------------|\n"
    
    # 构建表格内容
    rows = []
    for model, data in sorted(results, key=lambda x: x[1]["blurb_score"], reverse=True):
        task_scores = [f"{data['task_scores'].get(task, 0):.2f}%" for task in task_order]
        row = f"| {model} | " + " | ".join(task_scores) + f" | {data['blurb_score']}% |"
        rows.append(row)
    
    # 写入文件
    with open(output_file, "w") as f:
        f.write("# BLURB Benchmark Results\n\n")
        f.write(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        f.write(header)
        f.write(separator)
        f.write("\n".join(rows))

def process_results_folder(input_dir, output_file):
    """处理整个结果目录"""
    results = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt") and "BLURB_Score" in filename:
            try:
                model_name = filename.replace("-BLURB_Score.txt", "")
                file_path = os.path.join(input_dir, filename)
                metrics = parse_metrics_file(file_path)
                score_data = calculate_blurb_score(metrics)
                results.append((model_name, score_data))
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    generate_markdown_report(results, output_file)

if __name__ == "__main__":
    parser = ArgumentParser(description="BLURB Benchmark Reporter")
    parser.add_argument("--input_dir", required=True, help="Path to results folder")
    parser.add_argument("--output", default="blurb_report.md", help="Output file name")
    
    args = parser.parse_args()
    
    process_results_folder(
        input_dir=args.input_dir,
        output_file=args.output
    )