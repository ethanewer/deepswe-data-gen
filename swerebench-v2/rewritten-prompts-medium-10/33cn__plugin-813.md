# 33cn__plugin-813

- repo: 33cn/plugin
- language: go
- difficulty: medium

## Rewritten Prompt

优化平行链空块间隔的配置方式，使其更简单。现在配置应支持按单个 key-value 项来表达空块间隔规则，并能将这些配置解析为按生效高度排序的区间规则。

当未提供任何配置时，应使用默认规则：从高度 0 开始，采用默认空块间隔。每条配置都应能被解析为“起始高度:间隔”形式；起始高度必须是非负整数，间隔必须是正整数。若配置无法拆分为两个整数，或间隔不合法，应返回错误。

解析后的规则需要保持按起始高度递增，且各条规则的起始高度必须严格递增。最终得到的结构应能用于平行链共识模块按高度决定空块生成间隔。

## Preserved Requirements

- 支持将平行链空块间隔配置改为更简单的单个 key-value 形式。
- 空配置时应使用默认规则：起始高度为 0，间隔为默认空块间隔。
- 每条配置项应按“起始高度:间隔”形式解析。
- 起始高度必须是非负整数。
- 间隔必须是正整数。
- 无法拆分为两个整数或解析失败时应返回错误。
- 解析结果应按起始高度排序。
- 解析结果中的起始高度必须严格递增。
- 解析后的规则用于平行链共识模块决定空块生成间隔。

## Removed Noise

- 原始中文标题中的“优化”表述
- 关于批量读写 toml 配置文件支持复杂的背景说明
- 任何实现位置、函数名、类型名的显式引用
- 接口说明中的输入输出签名细节
- “tests”“PR”“patch”等测试或提交流程相关信息
- 路径与文件位置提示
- 外部 URL 或仓库元数据

## Risk Notes

- 原始描述未明确说明配置项是否允许重复高度以外的额外约束，只保留了“严格递增”的要求。
- 默认间隔的具体数值未在原始需求中给出，只能保留为“默认空块间隔”。

## Original Prompt

平行链空块间隔参数配置优化
之前空块间隔的配置在批量读写toml配置文件时候支持比较复杂，改为简单的单kv 类型

## Original Interface

Function: parseEmptyBlockInterval(cfg []string) ([]*emptyBlockInterval, error)
Location: plugin/consensus/para/para.go
Inputs: 
  - cfg []string – each element must be in the “startHeight:interval” form; an empty slice yields a single default interval (startHeight = 0, interval = defaultEmptyBlockInterval).
Outputs: 
  - []*emptyBlockInterval – slice of parsed interval objects ordered by startHeight.  
  - error – returned if any string cannot be split into two integers, if parsing fails, or if an interval value ≤ 0.
Description: Parses the empty‑block interval configuration strings supplied in the consensus sub‑config into structured interval entries, applying a default when the list is empty and ensuring the sequence is monotonic.

Type: emptyBlockInterval
Location: plugin/consensus/para/para.go
Fields:
  - startHeight int64 – block height at which this empty‑block interval becomes active (must be ≥ 0 and strictly increasing across entries).
  - interval int64 – number of main‑chain blocks between generated empty blocks for the parallel chain (must be > 0).
Description: Represents a single empty‑block interval rule used by the para consensus module to decide when to emit empty blocks on the parallel chain.
