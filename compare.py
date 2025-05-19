import sys
import numpy as np

def read_feature_vector(filename):
    """读取 txt 文件中的特征向量（支持逗号分隔、逗号+换行符、仅换行符分隔）"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if not lines:
                raise ValueError(f"File {filename} is empty")
            
            # 调试：打印文件原始内容
            print(f"Raw content of {filename}:")
            print(lines[:5] + (["..."] if len(lines) > 5 else []))  # 打印前5行以免过多输出
            
            # 处理所有格式
            values = []
            for line in lines:
                # 移除行首尾空白字符
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                # 检查是否包含逗号
                if ',' in line:
                    # 按逗号分割（支持逗号+换行符或单行逗号分隔）
                    line_values = [v.strip() for v in line.split(',') if v.strip()]
                    values.extend(line_values)
                else:
                    # 仅换行符分隔，直接添加整行
                    values.append(line)
            
            if not values:
                raise ValueError(f"No valid values found in {filename}")
            
            # 转换为浮点数
            feature = []
            for i, val in enumerate(values):
                try:
                    feature.append(float(val))
                except ValueError:
                    raise ValueError(f"Invalid float value at position {i+1} in {filename}: '{val}'")
            
            return np.array(feature, dtype=np.float32)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found")
    except Exception as e:
        raise Exception(f"Error processing {filename}: {str(e)}")

def compute_cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    if len(vec1) != len(vec2):
        raise ValueError(f"Feature vectors have different lengths: {len(vec1)} vs {len(vec2)}")
    
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    # 计算 L2 范数
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        raise ValueError("One or both vectors have zero norm, cannot compute cosine similarity")
    
    return dot_product / (norm1 * norm2)

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <feature_file1.txt> <feature_file2.txt>")
        sys.exit(1)
    
    try:
        # 读取两个特征向量
        print(f"Reading {sys.argv[1]}...")
        vec1 = read_feature_vector(sys.argv[1])
        print(f"Reading {sys.argv[2]}...")
        vec2 = read_feature_vector(sys.argv[2])
        
        # 打印向量长度以便调试
        print(f"Vector 1 length: {len(vec1)}")
        print(f"Vector 2 length: {len(vec2)}")
        
        # 计算余弦相似度
        similarity = compute_cosine_similarity(vec1, vec2)
        
        # 输出结果，保留 6 位小数
        print(f"Cosine Similarity: {similarity:.6f}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()