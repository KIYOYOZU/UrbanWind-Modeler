# UrbanWind-Modeler

用于城市风场可视化的 3D 建筑块体建模脚本。支持从参考底图交互采集建筑中心点，基于邻域密度自动估算建筑高度，并渲染生成 3D 城市体块图。

## 功能特性
- 参考底图交互采集建筑中心点
- KNN 密度驱动的高度分配
- 简洁光照+阴影的 3D 渲染
- 结果与数据存档，便于复用

## 目录结构
```
.
├── input/                 # 参考底图（png/jpg）
├── output/                # 生成结果与数据
├── urban_mesh_final.py    # 主脚本
├── requirements.txt       # 依赖
└── README.md
```

## 环境要求
- Python 3.9+
- numpy
- matplotlib

## 安装
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 快速开始
```bash
python urban_mesh_final.py
```
如果 `input/` 下存在图片，将自动进入交互采集流程；否则直接使用内置示例建筑生成 3D 模型。

## 交互采集流程
1. 点击圆心  
2. 点击圆周上的一点  
3. 点击一个建筑的长宽对角线两点（用于估计建筑尺寸）  
4. 左键点击建筑中心点，右键撤销，回车结束  

采集完成后数据会写入 `output/building_data.json`，下次运行可直接复用。

## 命令行参数
```bash
python urban_mesh_final.py --input input/your_image.png --output output/city_3d_model.png --seed 42
```
- `--input`：参考底图路径（png/jpg/jpeg）
- `--output`：输出图片路径
- `--seed`：随机种子（控制高度扰动）

## 输出文件
- `output/city_3d_model.png`：3D 渲染结果图
- `output/building_data.json`：建筑中心点与圆参数
- `output/building_stats.txt`：建筑数量与高度统计

## 配置项说明（DEFAULT_CONFIG）
在 `urban_mesh_final.py` 中可调整：
- `h_min` / `h_max`：高度范围
- `height_scale`：整体高度缩放
- `knn_k`：密度计算的邻域大小
- `view_elev` / `view_azim` / `view_roll`：视角参数
- `ground_color` / `building_color` / `edge_color`：配色
- `dpi` / `figsize`：渲染质量

## 常见问题
- 窗口不弹出：确认已安装 GUI 后端（Tk/Qt），并在本地桌面环境运行。
- 点击无响应：确保 matplotlib 窗口处于前台并获得焦点。
- 画面太“平”：适当增大 `view_elev` 或调整 `view_azim`。
