import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 配置matplotlib支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 强制使用交互式后端，确保窗口能弹出
try:
    import matplotlib
    # 优先尝试常用交互后端
    for backend in ["TkAgg", "Qt5Agg", "MacOSX", "QtAgg"]:
        try:
            matplotlib.use(backend)
            break
        except:
            continue
except:
    pass


BUILDINGS = [
    {"center": [130.5, -78.5], "size": [27.0, 29.0]},
    {"center": [-169.0, 5.0], "size": [26.0, 30.0]},
    {"center": [-46.0, 40.0], "size": [26.0, 30.0]},
    {"center": [456.0, 40.0], "size": [26.0, 30.0]},
    {"center": [197.5, 114.0], "size": [27.0, 30.0]},
    {"center": [-35.0, 159.5], "size": [26.0, 29.0]},
    {"center": [-192.5, 245.5], "size": [27.0, 29.0]},
    {"center": [354.0, 245.5], "size": [26.0, 29.0]},
    {"center": [123.0, 275.0], "size": [26.0, 30.0]},
    {"center": [57.5, 288.0], "size": [27.0, 30.0]},
    {"center": [170.0, 302.5], "size": [26.0, 29.0]},
    {"center": [113.5, 327.0], "size": [27.0, 30.0]},
    {"center": [-294.0, 341.0], "size": [26.0, 30.0]},
    {"center": [582.0, 341.0], "size": [26.0, 30.0]},
    {"center": [55.5, 353.5], "size": [27.0, 29.0]},
    {"center": [93.5, 366.0], "size": [27.0, 30.0]},
    {"center": [146.5, 379.0], "size": [27.0, 30.0]},
    {"center": [79.0, 405.5], "size": [26.0, 29.0]},
    {"center": [328.0, 425.5], "size": [26.0, 29.0]},
    {"center": [-64.5, 443.5], "size": [27.0, 29.0]},
    {"center": [-110.0, 457.5], "size": [26.0, 29.0]},
    {"center": [-30.5, 483.0], "size": [27.0, 30.0]},
    {"center": [-69.5, 492.5], "size": [27.0, 29.0]},
    {"center": [-113.0, 507.0], "size": [26.0, 30.0]},
    {"center": [501.5, 525.0], "size": [27.0, 30.0]},
    {"center": [-81.0, 533.5], "size": [26.0, 29.0]},
    {"center": [-41.0, 533.5], "size": [26.0, 29.0]},
    {"center": [281.0, 543.0], "size": [26.0, 30.0]},
    {"center": [-1.0, 558.0], "size": [26.0, 30.0]},
    {"center": [161.5, 635.5], "size": [27.0, 29.0]},
    {"center": [302.0, 662.0], "size": [26.0, 30.0]},
    {"center": [-62.0, 688.5], "size": [26.0, 29.0]},
]

CIRCLE_RADIUS = 842.0
CIRCLE_CENTER = (0.0, 0.0)

DEFAULT_CONFIG = {
    "output_path": "output/city_3d_model.png",
    "data_path": "output/building_data.json",
    "knn_k": 5,
    "h_min": 10.0,
    "h_max": 30.0,
    "height_scale": 1,
    "random_seed": 42,
    "view_elev": 25,
    "view_azim": 45,
    "view_roll": 0,
    "view_margin_ratio": 0.5,
    "ground_radius_ratio": 1.05,
    "ground_color": "#D5CCBF",
    "dpi": 300,
    "figsize": (12, 12),
    "axis_off": True,
    "use_height_color": False,
    "building_color": "#B0B0B0",
    "edge_color": "black",
    "edge_alpha": 0.3,
    "write_stats": True,
    "building_size": (26.0, 30.0),
}


def _ginput_one(fig, ax, title):
    ax.set_title(title)
    fig.canvas.draw()
    plt.pause(0.01)
    pts = plt.ginput(1, timeout=-1)
    if not pts:
        return None
    x, y = pts[0]
    ax.plot([x], [y], marker="o", color="cyan", markersize=6)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)
    return x, y


def collect_buildings_interactive(image_path):
    img = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    plt.axis("off")
    plt.ion()

    center = _ginput_one(fig, ax, "第1步：点击圆心")
    if center is None:
        plt.close(fig)
        raise RuntimeError("未检测到圆心点击。")

    edge = _ginput_one(fig, ax, "第2步：点击圆周上一点")
    if edge is None:
        plt.close(fig)
        raise RuntimeError("未检测到圆周点击。")

    cx, cy = center
    ex, ey = edge
    radius = float(np.hypot(ex - cx, ey - cy))

    circle_patch = Circle((cx, cy), radius, fill=False, color="#1f77b4", linewidth=2)
    ax.add_patch(circle_patch)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

    # 第3步：通过点击两点确定建筑尺寸
    ax.set_title("第3步：点击一个建筑的长宽对角线（两点）以确定尺寸")
    fig.canvas.draw()
    plt.pause(0.01)
    size_pts = plt.ginput(2, timeout=-1)
    if len(size_pts) < 2:
        plt.close(fig)
        raise RuntimeError("未采集到建筑尺寸点喵。")
    
    p1, p2 = size_pts
    bw = abs(p2[0] - p1[0])
    bd = abs(p2[1] - p1[1])
    print(f"喵～ 自动计算建筑尺寸: 宽度={bw:.2f}, 深度={bd:.2f}")

    # 在图上画一个小框示意一下采集到的尺寸
    rect = plt.Rectangle((min(p1[0], p2[0]), min(p1[1], p2[1])), bw, bd,
                        fill=False, color="yellow", linestyle="--", linewidth=1)
    ax.add_patch(rect)
    
    ax.set_title("第4步：左键点击建筑中心，右键撤销，回车(Enter)结束")
    fig.canvas.draw()
    plt.pause(0.01)
    # mouse_add=1 (左键), mouse_pop=3 (右键), mouse_stop=2 (中键)
    pts = plt.ginput(n=-1, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=2)

    if not pts:
        plt.close(fig)
        raise RuntimeError("未采集到任何建筑点。")

    buildings = []
    for x, y in pts:
        x_3d = x - cx
        y_3d = cy - y
        buildings.append({"center": [float(x_3d), float(y_3d)], "size": [float(bw), float(bd)]})

    plt.close(fig)
    return buildings, radius, (0.0, 0.0)


def calculate_density_knn(positions, k=5):
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return np.array([])
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist_sorted = np.sort(dist, axis=1)
    k_eff = min(k + 1, dist_sorted.shape[1])
    if k_eff <= 1:
        avg_dist = dist_sorted[:, 0]
    else:
        avg_dist = np.mean(dist_sorted[:, 1:k_eff], axis=1)
    return np.where(avg_dist > 0, 1.0 / avg_dist, 0.0)


def assign_heights(densities, h_min=3.0, h_max=25.0, random_seed=None):
    densities = np.asarray(densities, dtype=float)
    if densities.size == 0:
        return np.array([])
    d_min = float(np.min(densities))
    d_max = float(np.max(densities))
    if d_max <= d_min:
        norm = np.zeros_like(densities)
    else:
        norm = (densities - d_min) / (d_max - d_min)
    heights = h_min + (h_max - h_min) * np.sqrt(norm)
    rng = np.random.default_rng(random_seed)
    heights *= rng.uniform(0.9, 1.1, size=heights.shape[0])
    return heights


def create_building_box(x, y, z, width, depth, height):
    vertices = [
        [x, y, z],
        [x + width, y, z],
        [x + width, y + depth, z],
        [x, y + depth, z],
        [x, y, z + height],
        [x + width, y, z + height],
        [x + width, y + depth, z + height],
        [x, y + depth, z + height],
    ]
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]
    return vertices, faces


def _compute_face_normal(face_vertices):
    v0 = np.array(face_vertices[0], dtype=float)
    v1 = np.array(face_vertices[1], dtype=float)
    v2 = np.array(face_vertices[2], dtype=float)
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n)
    if norm == 0:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def calculate_face_brightness(face_normal, light_dir=(1.0, 1.0, 2.0)):
    light = np.array(light_dir, dtype=float)
    light = light / np.linalg.norm(light)
    brightness = float(np.dot(face_normal, light))
    return max(0.0, 0.4 + 0.6 * brightness)


def get_building_color(height, max_height, cmap_name="YlGnBu"):
    if max_height <= 0:
        ratio = 0.0
    else:
        ratio = max(0.0, min(1.0, height / max_height))
    cmap = cm.get_cmap(cmap_name)
    return np.array(cmap(ratio))


def _resolve_color(color, default="#B0B0B0"):
    if color is None:
        return np.array(mcolors.to_rgba(default))
    if isinstance(color, str):
        return np.array(mcolors.to_rgba(color))
    if isinstance(color, (list, tuple)) and len(color) in (3, 4):
        return np.array(color, dtype=float)
    return np.array(mcolors.to_rgba(default))


def draw_building_with_shading(
    ax,
    vertices,
    faces,
    height,
    max_height,
    light_dir=(1.0, 1.0, 2.0),
    draw_edges=True,
    edge_color="black",
    edge_alpha=0.3,
    include_bottom=False,
    use_height_color=True,
    base_color=None,
    sort_zpos=None,
):
    face_vertices = []
    face_colors = []
    for idx, face in enumerate(faces):
        if not include_bottom and idx == 0:
            continue
        verts = [vertices[i] for i in face]
        normal = _compute_face_normal(verts)
        brightness = calculate_face_brightness(normal, light_dir)
        if use_height_color:
            base_color = get_building_color(height, max_height)
        else:
            base_color = _resolve_color(base_color)
        color = base_color.copy()
        color[:3] = np.clip(base_color[:3] * brightness, 0.0, 1.0)
        face_vertices.append(verts)
        face_colors.append(color)

    if draw_edges:
        edge_rgba = _resolve_color(edge_color, default="black")
        edge_rgba[3] = float(edge_alpha)
        edgecolors = edge_rgba
        linewidths = 0.3
    else:
        edgecolors = "none"
        linewidths = 0.0

    poly = Poly3DCollection(
        face_vertices,
        facecolors=face_colors,
        edgecolors=edgecolors,
        linewidths=linewidths,
    )
    if sort_zpos is not None and hasattr(poly, "set_sort_zpos"):
        poly.set_sort_zpos(float(sort_zpos))
    ax.add_collection3d(poly)


def draw_ground_circle(ax, center, radius, color="#E8E8E8", alpha=0.9, resolution=100, z_offset=-1e-3, zsort="min", zorder=0, sort_zpos=-1e6):
    cx, cy = center
    theta = np.linspace(0.0, 2.0 * np.pi, resolution)
    r = np.linspace(0.0, radius, int(resolution / 2))
    t_grid, r_grid = np.meshgrid(theta, r)
    x = cx + r_grid * np.cos(t_grid)
    y = cy + r_grid * np.sin(t_grid)
    z = np.full_like(x, float(z_offset))
    surface = ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=True, linewidth=0, zorder=zorder)
    if zsort and hasattr(surface, "set_zsort"):
        surface.set_zsort(zsort)
    if sort_zpos is not None and hasattr(surface, "set_sort_zpos"):
        surface.set_sort_zpos(float(sort_zpos))


def save_statistics(output_dir, buildings, heights, circle_radius):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "building_stats.txt")
    heights = np.asarray(heights, dtype=float)
    lines = [
        f"buildings: {len(buildings)}",
        f"height_min: {heights.min():.3f}",
        f"height_max: {heights.max():.3f}",
        f"height_mean: {heights.mean():.3f}",
        f"circle_radius: {circle_radius:.3f}",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_model(config, buildings=None, circle_radius=None, circle_center=None):
    print(f"喵～ 开始构建 3D 城市模型，建筑物数量: {len(buildings if buildings else BUILDINGS)}")
    if buildings is None:
        buildings = BUILDINGS
    if circle_radius is None:
        circle_radius = CIRCLE_RADIUS
    if circle_center is None:
        circle_center = CIRCLE_CENTER

    positions = np.array([b["center"] for b in buildings], dtype=float)
    densities = calculate_density_knn(positions, k=config["knn_k"])
    heights = assign_heights(
        densities,
        h_min=config["h_min"],
        h_max=config["h_max"],
        random_seed=config["random_seed"],
    )
    heights = heights * float(config["height_scale"])

    fig = plt.figure(figsize=config["figsize"], dpi=config["dpi"])
    ax = fig.add_subplot(111, projection="3d")
    # 使用正交投影，避免透视变形
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("ortho")
    if config["axis_off"]:
        ax.set_axis_off()
    else:
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    ground_radius = circle_radius * float(config["ground_radius_ratio"])
    draw_ground_circle(ax, center=circle_center, radius=ground_radius, color=config["ground_color"])

    for i, building in enumerate(buildings):
        cx, cy = building["center"]
        w, d = building["size"]
        x = cx - w / 2.0
        y = cy - d / 2.0
        vertices, faces = create_building_box(x, y, 0.0, w, d, float(heights[i]))
        draw_building_with_shading(
            ax,
            vertices,
            faces,
            height=float(heights[i]),
            max_height=float(config["h_max"]),
            use_height_color=bool(config["use_height_color"]),
            base_color=config["building_color"],
            edge_color=config["edge_color"],
            edge_alpha=config["edge_alpha"],
        )

    view_radius = circle_radius * 1.5
    ax.set_xlim([-view_radius, view_radius])
    ax.set_ylim([-view_radius, view_radius])
    ax.set_zlim([0, circle_radius * 0.5])

    ax.view_init(
        elev=config["view_elev"],
        azim=config["view_azim"],
        roll=config["view_roll"],
    )

    os.makedirs(os.path.dirname(config["output_path"]), exist_ok=True)
    plt.tight_layout()
    print(f"正在渲染并保存模型到: {config['output_path']} ... (这可能需要一点时间喵)")
    fig.savefig(
        config["output_path"],
        dpi=config["dpi"],
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print("模型保存成功喵！o(*￣︶￣*)o")

    if config["write_stats"]:
        output_dir = os.path.dirname(config["output_path"])
        save_statistics(output_dir, buildings, heights, circle_radius)


def parse_args():
    parser = argparse.ArgumentParser(description="3D city building layout modeler")
    parser.add_argument("--input", default=None, help="Interactive reference image path")
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument("--seed", default=None, type=int, help="Random seed override")
    return parser.parse_args()

def main():
    args = parse_args()
    config = dict(DEFAULT_CONFIG)
    if args.output:
        config["output_path"] = args.output
    if args.seed is not None:
        config["random_seed"] = args.seed

    data_path = config["data_path"]
    
    # 检查是否有存档
    if os.path.exists(data_path):
        print(f"喵～ 发现历史采集数据: {data_path}")
        try:
            choice = input("是要 [1] 加载历史数据 还是 [2] 重新开始采集喵？(输入1或2): ").strip()
            if choice == "1":
                with open(data_path, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                print("存档加载成功喵！开始直接生成模型...")
                build_model(config,
                           buildings=saved_data["buildings"],
                           circle_radius=saved_data["circle_radius"],
                           circle_center=saved_data["circle_center"])
                return
        except EOFError:
            print("检测到非交互环境，默认跳过存档加载喵～")
        except Exception as e:
            print(f"存档加载失败了喵: {e}，将尝试进入采集流程喵...")

    input_img = args.input
    # 如果没传参数，自动寻找 input 文件夹下的 png
    if not input_img:
        input_dir = "input"
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if files:
                input_img = os.path.join(input_dir, files[0])
                print(f"喵～ 自动检测到底图: {input_img}")

    if input_img:
        print("进入交互模式，请在弹出的窗口中进行采集喵～")
        try:
            buildings, circle_radius, circle_center = collect_buildings_interactive(input_img)
            
            # 采集完顺便存个档喵
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump({
                    "buildings": buildings,
                    "circle_radius": circle_radius,
                    "circle_center": circle_center
                }, f, indent=4)
            print(f"采集数据已存档到: {data_path} 喵～")
            
            build_model(config, buildings=buildings, circle_radius=circle_radius, circle_center=circle_center)
        except Exception as e:
            print(f"交互模式运行失败了喵: {e}")
            print("退回到默认模型生成模式喵...")
            build_model(config)
    else:
        print("未发现底图，直接生成默认城市模型喵～")
        build_model(config)


if __name__ == "__main__":
    main()
