import re

# 定义一个函数来提取所需的数据
def extract_data(line):
    # 修改正则表达式以包含iter的提取
    pattern = r"\[validate\]: \[iter (\d+)\], \[loss ([0-9\.]+)\] \[PSNR ([0-9\.]+)\] \[SSIM ([0-9\.]+)\] \[MSE ([0-9\.]+)\]"
    match = re.search(pattern, line)
    if match:
        return match.groups()
    return None

# 读取原始文件并写入新文件
def process_file(input_file_path, output_file_path):
    max_psnr = 0
    max_ssim = 0
    min_mse = float('inf')  # 初始化为无穷大
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    
    extracted_data = []
    for line in lines:
        data = extract_data(line)
        if data:
            # 将iter的数据作为列表的第一个元素
            extracted_data.append(data)
            if float(data[2]) > max_psnr:
                max_psnr = float(data[2])
            if float(data[3]) > max_ssim:
                max_ssim = float(data[3])
            if float(data[4]) < min_mse:
                min_mse = float(data[4])
    with open(output_file_path, 'w') as file:
        for data in extracted_data:
            # 写入iter的数据和其他数据
            file.write(f"iter: {data[0]}, loss: {data[1]}, PSNR: {data[2]}, SSIM: {data[3]}, MSE: {data[4]}\n")
        file.write("\nMaximum PSNR: {:.4f}\n".format(max_psnr))
        file.write("Maximum SSIM: {:.4f}\n".format(max_ssim))
        file.write("Minimum MSE: {:.4f}\n".format(min_mse))

# 调用函数处理文件
input_file_path = 'duo3.txt'  # 替换为你的输入文件路径
output_file_path = 'duo3_out.txt'  # 替换为你想要保存输出的文件路径
process_file(input_file_path, output_file_path)