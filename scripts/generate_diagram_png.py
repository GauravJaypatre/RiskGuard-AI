"""
RiskGuard AI — Presentation-Ready 16:9 Architecture Diagram Generator
Produces an ultra-sharp, high-contrast 1920x1080 PNG with dark glassmorphic styling,
custom vector icons, and bold left-to-right flow arrows.
"""

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080

# 1. Base Dark Canvas (#0c0f17)
base = Image.new("RGB", (W, H), (12, 15, 23))

# 2. Add subtle radial background glow
glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
glow_draw = ImageDraw.Draw(glow_layer)

def add_glow(cx, cy, r_max, color, max_alpha=35):
    r_c, g_c, b_c = color
    for r in range(r_max, 0, -8):
        alpha = int(max_alpha * (1.0 - r / r_max) ** 2)
        glow_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(r_c, g_c, b_c, alpha))

add_glow(350, 540, 500, (59, 130, 246), 40)   # Blue
add_glow(960, 540, 450, (245, 158, 11), 35)   # Amber
add_glow(1570, 540, 500, (16, 185, 129), 40)  # Green

base.paste(Image.alpha_composite(Image.new("RGBA", (W, H), (12, 15, 23, 255)), glow_layer).convert("RGB"))
draw = ImageDraw.Draw(base)

# 3. Typography
font_title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 36)
font_subtitle = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
font_badge = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 13)
font_layer_tag = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 13)
font_layer_title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 26)
font_layer_sub = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14)
font_comp_title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 16)
font_comp_sub = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
font_arrow = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 14)
font_caption = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 16)

# 4. Header Section
# Brand Shield Icon
draw.rounded_rectangle([70, 46, 124, 100], radius=14, fill=(20, 30, 50), outline=(59, 130, 246), width=2)
shield_pts = [(97, 58), (112, 65), (109, 82), (97, 91), (85, 82), (82, 65)]
draw.polygon(shield_pts, fill=(59, 130, 246))
draw.polygon([(97, 61), (107, 66), (105, 80), (97, 87)], fill=(147, 197, 253))

draw.text((142, 46), "RiskGuard AI", font=font_title, fill=(255, 255, 255))
draw.text((144, 90), "Autonomous Risk Intelligence & Fraud Investigation Architecture", font=font_subtitle, fill=(148, 163, 184))

# Buildathon Tag
tag_text = "RAZORPAY AI BUILDATHON  •  AI RISK MANAGER"
tag_w = int(draw.textlength(tag_text, font=font_badge))
tag_x = W - 70 - tag_w - 40
draw.rounded_rectangle([tag_x, 56, W - 70, 92], radius=18, fill=(20, 28, 44), outline=(71, 85, 105), width=1)
draw.ellipse([tag_x + 16, 70, tag_x + 24, 78], fill=(59, 130, 246))
draw.text((tag_x + 34, 65), tag_text, font=font_badge, fill=(203, 213, 225))

# Header Divider
draw.line([(70, 118), (W - 70, 118)], fill=(38, 48, 68), width=1)

# 5. Three Columns Setup
card_w = 490
card_h = 750
card_y = 160

c1_x = 70
c2_x = 715
c3_x = 1360

layers = [
    {
        "x": c1_x,
        "tag": "LAYER 1",
        "title": "DETECTION",
        "subtitle": "Statistical & Behavioral Threat Discovery",
        "accent": (59, 130, 246),
        "tag_bg": (26, 46, 82),
        "tag_fg": (96, 165, 250),
        "components": [
            {
                "title": "XGBoost Real-Time Scoring",
                "desc": "Real-time transaction inference with cost-asymmetric probability calibration (PR-AUC 0.7913)",
                "icon_type": "bolt"
            },
            {
                "title": "Merchant Behavioral Baselining",
                "desc": "Continuous daily volume, risk, and failure rate anomaly tracking (z-score deviation ≥2.0σ)",
                "icon_type": "bars"
            },
            {
                "title": "Graph Relationship Engine",
                "desc": "Bipartite customer-device clustering to uncover threshold-aware distributed abuse rings",
                "icon_type": "graph"
            }
        ]
    },
    {
        "x": c2_x,
        "tag": "LAYER 2",
        "title": "EXPLAINABILITY",
        "subtitle": "Financial Impact & Actionable Interpretability",
        "accent": (245, 158, 11),
        "tag_bg": (70, 48, 18),
        "tag_fg": (251, 191, 36),
        "components": [
            {
                "title": "Financial Exposure Estimation",
                "desc": "Potential exposure (threshold ≥ 0.33) vs high-confidence exposure (≥ 0.80) — 'identified, not claimed prevented'",
                "icon_type": "coin"
            },
            {
                "title": "Counterfactual Engine",
                "desc": "Nearest minimal feature perturbations required to flip high-risk alert below decision boundary",
                "icon_type": "arrows"
            }
        ]
    },
    {
        "x": c3_x,
        "tag": "LAYER 3",
        "title": "AUTONOMOUS AGENT",
        "subtitle": "Policy-Bounded Autonomous Investigation",
        "accent": (16, 185, 129),
        "tag_bg": (20, 56, 44),
        "tag_fg": (52, 211, 153),
        "components": [
            {
                "title": "Gemini Investigation Agent",
                "desc": "Autonomous multi-turn reasoning and dynamic tool orchestration with honest live/mock fallback",
                "icon_type": "bot"
            },
            {
                "title": "10-Tool Operational Toolkit",
                "desc": "Transaction, customer, device, velocity, baseline, cluster, and policy authorization lookups",
                "icon_type": "tools"
            },
            {
                "title": "Case Creation & Audit Trail",
                "desc": "Structured case synthesis with complete evidence log, cluster mapping, and policy action output",
                "icon_type": "case"
            }
        ]
    }
]

# Vector Icon Drawers
def draw_icon(draw, itype, cx, cy, color):
    r, g, b = color
    if itype == "bolt":
        # Lightning Bolt
        pts = [(cx - 3, cy - 14), (cx + 8, cy - 14), (cx - 1, cy - 2), (cx + 7, cy - 2), (cx - 7, cy + 14), (cx - 3, cy + 2), (cx - 9, cy + 2)]
        draw.polygon(pts, fill=(r, g, b))
    elif itype == "bars":
        # 3 Chart Bars
        draw.rounded_rectangle([cx - 13, cy + 2, cx - 7, cy + 12], radius=2, fill=(r, g, b))
        draw.rounded_rectangle([cx - 3, cy - 6, cx + 3, cy + 12], radius=2, fill=(r, g, b))
        draw.rounded_rectangle([cx + 7, cy - 12, cx + 13, cy + 12], radius=2, fill=(r, g, b))
    elif itype == "graph":
        # Graph Nodes
        pts = [(cx - 8, cy - 8), (cx + 8, cy - 8), (cx, cy + 8)]
        draw.line([pts[0], pts[1]], fill=(r, g, b), width=2)
        draw.line([pts[1], pts[2]], fill=(r, g, b), width=2)
        draw.line([pts[2], pts[0]], fill=(r, g, b), width=2)
        for p in pts:
            draw.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=(r, g, b))
    elif itype == "coin":
        # Coin / Exposure
        draw.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], outline=(r, g, b), width=2)
        draw.line([(cx - 5, cy - 4), (cx + 5, cy - 4)], fill=(r, g, b), width=2)
        draw.line([(cx - 5, cy), (cx + 4, cy)], fill=(r, g, b), width=2)
        draw.line([(cx - 2, cy - 8), (cx - 2, cy + 8)], fill=(r, g, b), width=2)
    elif itype == "arrows":
        # Two arrows in circle
        draw.arc([cx - 12, cy - 12, cx + 12, cy + 12], start=30, end=150, fill=(r, g, b), width=2)
        draw.arc([cx - 12, cy - 12, cx + 12, cy + 12], start=210, end=330, fill=(r, g, b), width=2)
        draw.polygon([(cx + 9, cy - 8), (cx + 14, cy - 2), (cx + 6, cy - 3)], fill=(r, g, b))
        draw.polygon([(cx - 9, cy + 8), (cx - 14, cy + 2), (cx - 6, cy + 3)], fill=(r, g, b))
    elif itype == "bot":
        # AI Agent Chip / Head
        draw.rounded_rectangle([cx - 11, cy - 9, cx + 11, cy + 9], radius=4, outline=(r, g, b), width=2)
        draw.ellipse([cx - 6, cy - 3, cx - 2, cy + 1], fill=(r, g, b))
        draw.ellipse([cx + 2, cy - 3, cx + 6, cy + 1], fill=(r, g, b))
        draw.line([(cx - 4, cy + 5), (cx + 4, cy + 5)], fill=(r, g, b), width=2)
        draw.line([(cx, cy - 9), (cx, cy - 13)], fill=(r, g, b), width=2)
        draw.ellipse([cx - 2, cy - 15, cx + 2, cy - 11], fill=(r, g, b))
    elif itype == "tools":
        # Wrench / Toolbox
        draw.rounded_rectangle([cx - 11, cy - 8, cx + 11, cy + 10], radius=3, outline=(r, g, b), width=2)
        draw.rounded_rectangle([cx - 5, cy - 12, cx + 5, cy - 8], radius=2, outline=(r, g, b), width=1)
        draw.line([(cx - 6, cy + 1), (cx + 6, cy + 1)], fill=(r, g, b), width=2)
    elif itype == "case":
        # Case / Audit document
        draw.rounded_rectangle([cx - 9, cy - 12, cx + 9, cy + 12], radius=2, outline=(r, g, b), width=2)
        draw.line([(cx - 5, cy - 5), (cx + 5, cy - 5)], fill=(r, g, b), width=1)
        draw.line([(cx - 5, cy), (cx + 5, cy)], fill=(r, g, b), width=1)
        draw.line([(cx - 5, cy + 5), (cx + 2, cy + 5)], fill=(r, g, b), width=1)

# Render Layer Cards
for l in layers:
    x = l["x"]
    acc = l["accent"]
    
    # Outer Card
    draw.rounded_rectangle([x, card_y, x + card_w, card_y + card_h], radius=24, fill=(18, 24, 38), outline=acc, width=2)
    
    # Top Tag
    tag_w = int(draw.textlength(l["tag"], font=font_layer_tag))
    draw.rounded_rectangle([x + 32, card_y + 28, x + 32 + tag_w + 24, card_y + 56], radius=14, fill=l["tag_bg"], outline=acc, width=1)
    draw.text((x + 44, card_y + 34), l["tag"], font=font_layer_tag, fill=l["tag_fg"])
    
    # Layer Title & Subtitle
    draw.text((x + 32, card_y + 70), l["title"], font=font_layer_title, fill=(255, 255, 255))
    draw.text((x + 32, card_y + 108), l["subtitle"], font=font_layer_sub, fill=(148, 163, 184))
    
    # Card Header Divider
    draw.line([(x + 32, card_y + 140), (x + card_w - 32, card_y + 140)], fill=(34, 46, 68), width=1)
    
    # Components
    comps = l["components"]
    num_comps = len(comps)
    
    comp_box_h = 160 if num_comps == 3 else 235
    comp_gap = 20 if num_comps == 3 else 40
    start_comp_y = card_y + 165
    
    for i, c in enumerate(comps):
        cy = start_comp_y + i * (comp_box_h + comp_gap)
        bx1 = x + 26
        bx2 = x + card_w - 26
        by1 = cy
        by2 = cy + comp_box_h
        
        # Component Box (Dark Solid Navy with High Contrast)
        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=16, fill=(26, 34, 52), outline=(48, 62, 90), width=1)
        
        # Left Accent Color Pill
        draw.rounded_rectangle([bx1 + 1, by1 + 6, bx1 + 6, by2 - 6], radius=3, fill=acc)
        
        # Icon Container
        icon_cx = bx1 + 42
        icon_cy = by1 + (comp_box_h // 2)
        draw.rounded_rectangle([icon_cx - 22, icon_cy - 22, icon_cx + 22, icon_cy + 22], radius=12, fill=(35, 46, 70), outline=acc, width=1)
        draw_icon(draw, c["icon_type"], icon_cx, icon_cy, acc)
        
        # Text Content
        tx = bx1 + 80
        tw_max = bx2 - tx - 20
        
        # Title
        title_y = by1 + 24 if num_comps == 3 else by1 + 50
        draw.text((tx, title_y), c["title"], font=font_comp_title, fill=(255, 255, 255))
        
        # Description wrapping
        words = c["desc"].split()
        lines = []
        cur_line = ""
        for w in words:
            test_line = cur_line + (" " if cur_line else "") + w
            if draw.textlength(test_line, font=font_comp_sub) < tw_max:
                cur_line = test_line
            else:
                lines.append(cur_line)
                cur_line = w
        if cur_line:
            lines.append(cur_line)
            
        desc_y = title_y + 32
        for l_idx, dl in enumerate(lines[:3]):
            draw.text((tx, desc_y + l_idx * 21), dl, font=font_comp_sub, fill=(160, 174, 192))

# 6. Connecting Arrows between Layers
def draw_connector(x1, x2, y_mid, label_text):
    cx = (x1 + x2) // 2
    lbl_w = int(draw.textlength(label_text, font=font_arrow))
    
    # Label Pill
    pw = lbl_w + 32
    ph = 32
    draw.rounded_rectangle([cx - pw // 2, y_mid - 40, cx + pw // 2, y_mid - 8], radius=16, fill=(30, 41, 59), outline=(71, 85, 105), width=1)
    draw.text((cx - lbl_w // 2, y_mid - 32), label_text, font=font_arrow, fill=(248, 250, 252))
    
    # Horizontal line
    line_y = y_mid + 6
    draw.line([(x1 + 8, line_y), (x2 - 14, line_y)], fill=(148, 163, 184), width=3)
    
    # Arrow head
    hx = x2 - 8
    head_pts = [(hx, line_y), (hx - 14, line_y - 8), (hx - 14, line_y + 8)]
    draw.polygon(head_pts, fill=(203, 213, 225))

draw_connector(c1_x + card_w, c2_x, card_y + 360, "flags")
draw_connector(c2_x + card_w, c3_x, card_y + 360, "investigates")

# 7. Bottom Caption Pill
cap_text = "Detection   →   Explainability   →   Autonomous Investigation"
cap_w = int(draw.textlength(cap_text, font=font_caption))
cap_x = (W - cap_w) // 2
draw.rounded_rectangle([cap_x - 36, H - 90, cap_x + cap_w + 36, H - 42], radius=24, fill=(18, 24, 38), outline=(51, 65, 85), width=1)

# Step 1: Detection
dot_y = H - 66
draw.ellipse([cap_x - 14, dot_y - 5, cap_x - 4, dot_y + 5], fill=(59, 130, 246))
draw.text((cap_x + 8, H - 76), "Detection", font=font_caption, fill=(96, 165, 250))

arr1_x = cap_x + int(draw.textlength("Detection", font=font_caption)) + 26
draw.text((arr1_x, H - 76), "→", font=font_caption, fill=(100, 116, 139))

# Step 2: Explainability
exp_x = arr1_x + 36
draw.ellipse([exp_x - 14, dot_y - 5, exp_x - 4, dot_y + 5], fill=(245, 158, 11))
draw.text((exp_x + 8, H - 76), "Explainability", font=font_caption, fill=(251, 191, 36))

arr2_x = exp_x + int(draw.textlength("Explainability", font=font_caption)) + 26
draw.text((arr2_x, H - 76), "→", font=font_caption, fill=(100, 116, 139))

# Step 3: Autonomous Investigation
inv_x = arr2_x + 36
draw.ellipse([inv_x - 14, dot_y - 5, inv_x - 4, dot_y + 5], fill=(16, 185, 129))
draw.text((inv_x + 8, H - 76), "Autonomous Investigation", font=font_caption, fill=(52, 211, 153))

# Save PNG output
output_path = "architecture_diagram.png"
base.save(output_path, "PNG", optimize=True)
print(f"High-resolution diagram generated successfully: {output_path} (1920x1080)")
