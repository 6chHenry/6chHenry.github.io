# 北欧旅行规划页设计

## Summary

在笔记页面新增 `Travel` 分类，并创建第一篇“北欧旅行规划”视觉灵感页。页面以 Stockholm 为 base，采用“地图主导 + 日程卡片”的方案：顶部用卡通北欧路线图展示 Stockholm、Oslo、Bergen、Helsinki、Tromsø 的空间关系；下方用 5 天主线卡片给出可执行旅行骨架；Helsinki 和 Tromsø 作为可选延伸，不硬塞进 5 天主线。

该页面是旅行规划灵感页，不是已经完成的游记。内容需要明确区分“主线可执行路线”和“未来可选支线”。

## Approved Direction

- 方案选择：方案 B，新增可复用 Astro 旅行规划组件。
- 页面定位：视觉灵感页，地图和路线氛围优先，实际交通可行性作为辅助说明。
- 旅行时长：5 天主线 + 可选延伸。
- 季节定位：通用规划版，不绑定冬季或夏季。
- 笔记目录：`docs/notes/Travel/`。

## Source Files

新增源内容：

- `docs/notes/Travel/北欧旅行规划.md`

新增源资产：

- `docs/notes/Travel/北欧旅行规划.assets/nordic-route.svg`

新增 Astro 组件：

- `site-next/src/components/travel/TravelRouteMap.astro`
- `site-next/src/components/travel/TravelItinerary.astro`
- `site-next/src/components/travel/TravelCityCard.astro`

新增样式：

- `site-next/src/styles/travel-planning.css`

生成内容仍由现有迁移流程处理，不直接编辑 `site-next/dist/`。

## Page Structure

页面按以下顺序组织：

1. Hero
   - 标题：北欧旅行规划。
   - 说明：从 Stockholm 出发的 5 天主线与可选延伸。
   - 标注：这是规划页，不是已完成游记。

2. Cartoon Route Map
   - 主导视觉。
   - 显示 Stockholm、Oslo、Bergen、Helsinki、Tromsø。
   - Stockholm 标记为 base。
   - 实线表示 5 天主线。
   - 虚线表示 Helsinki / Tromsø 可选延伸。
   - 点线表示慢旅行备选交通。

3. Five-Day Main Itinerary
   - Day 1: Stockholm
   - Day 2: Stockholm → Oslo
   - Day 3: Oslo → Bergen
   - Day 4: Bergen / fjord-style day trip
   - Day 5: Bergen → Stockholm 或 Oslo → Stockholm，作为返程缓冲

4. Optional Extensions
   - Helsinki：建议通过 Stockholm overnight ferry 或飞行，加 1-2 天。
   - Tromsø：建议通过 Oslo / Stockholm 飞行，加 2-3 天。

5. Transport Overview
   - 飞机：适合跨区域连接，尤其 Tromsø。
   - 火车：适合 Oslo → Bergen 或慢旅行版本。
   - 渡轮：适合 Stockholm → Helsinki。
   - 页面应避免暗示所有城市都适合塞进 5 天。

6. Planning Checklist
   - 预算。
   - 季节。
   - 申根与出入境。
   - 住宿城市。
   - 极光 / 峡湾优先级。

## Route Content

主线聚焦 Stockholm + Norway corridor：

- Stockholm 是 base 和出发点。
- Oslo 是中转兼城市体验。
- Bergen 是峡湾入口，也是主线的视觉重点。
- Helsinki 和 Tromsø 是独立支线，不进入 5 天主线。

页面文案应使用类似以下原则：

> 5 天主线聚焦 Stockholm + Norway corridor；Helsinki 和 Tromsø 是可选延伸，不建议硬塞进同一个 5 天行程。

## Component Design

### TravelRouteMap

职责：

- 渲染卡通路线图区域。
- 接收城市点位、路线线段和图例数据。
- 支持主线 / 延伸线 / 慢旅行备选三种线型。

实现约束：

- 使用 SVG 坐标，不做真实 GIS 投影。
- 目标是视觉表达路线关系，不追求地理精确。
- 首版静态渲染，不依赖客户端 JS。

### TravelItinerary

职责：

- 渲染 5 天主线卡片。
- 每张卡展示 day、route、transport、theme、reality check。

实现约束：

- 移动端单列。
- 桌面端可用网格或横向阶段布局。
- 内容必须可被 Pagefind 索引。

### TravelCityCard

职责：

- 渲染城市规划卡片。
- 展示城市定位、冬季亮点、夏季亮点、交通接入和主线 / 延伸标签。

实现约束：

- 可复用于未来旅行规划页。
- 不绑定北欧专有字段名。

## Data Shape

组件首版使用页面内静态数据或 Astro frontmatter 数据，不接外部 API。

建议结构：

```ts
type TravelStop = {
  id: string;
  name: string;
  country: string;
  role: "base" | "main" | "extension";
  days: string;
  transport: string;
  winter: string;
  summer: string;
  notes: string;
};

type ItineraryDay = {
  day: string;
  title: string;
  route: string;
  transport: string;
  theme: string;
  realisticNote: string;
};
```

数据量小，首版不需要 JSON 数据文件。若未来旅行规划页增多，再考虑抽成共享 data 模块。

## Visual Design

整体视觉使用北欧冷色系：

- 主色：深海蓝、冰蓝、松林绿。
- 点缀：极光紫、暖黄色窗口灯光。
- 地图：简化卡通 SVG。
- 卡片：半透明浅色规划卡片。

交互首版保持静态：

- hover 阴影或轻微位移可以用 CSS 实现。
- 不引入复杂 JS。
- 地图标记 hover 可有 CSS title/label，但不依赖交互才能理解内容。

## Navigation And Routing

新增 `docs/notes/Travel/北欧旅行规划.md` 后，现有 notes nav 扫描流程应自动生成：

- `/notes/category/travel/`
- `/notes/Travel/北欧旅行规划/`

`Travel` 目录名使用英文，与现有 `docs/essay/Travel` 保持一致。

## Search

页面正文需包含以下可搜索关键词：

- 北欧旅行
- Stockholm
- Oslo
- Bergen
- Helsinki
- Tromsø
- 斯德哥尔摩
- 卑尔根
- 奥斯陆
- 赫尔辛基
- 特罗姆瑟

Pagefind 应能搜到该页面。

## Verification Plan

实现后运行：

```powershell
cd F:\EECS498\6ch\site-next
npm run build
npm run preview -- --host 127.0.0.1 --port 4321
```

人工检查：

- Notes 导航出现 `Travel` 分类。
- `Travel` 分类下出现 `北欧旅行规划`。
- `/notes/Travel/北欧旅行规划/` 页面可访问。
- 地图 SVG 正常显示。
- 5 天主线卡片显示在地图下方。
- Helsinki / Tromsø 显示为可选延伸。
- 移动端布局不横向撑破。
- Pagefind 搜索 `北欧旅行`、`Stockholm`、`Bergen`、`Tromsø` 能返回该页面。
- `.superpowers/brainstorm/` 临时文件不提交。

## Out Of Scope

- 不做真实 GIS 地图。
- 不接地图 API。
- 不做预算自动计算。
- 不做航班实时查询。
- 不生成复杂客户端交互。
- 不把五个城市强行写成 5 天全覆盖的可执行路线。
