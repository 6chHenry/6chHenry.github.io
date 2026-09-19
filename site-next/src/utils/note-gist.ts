const CJK = /[\u4e00-\u9fff]/;
const TITLE_HINTS: Array<[RegExp, string]> = [
  [/docstring/i, '规范注释'],
  [/regex|regular/i, '正则表达式'],
  [/logging|\blog\b/i, '日志输出'],
  [/pandas/i, '表格数据'],
  [/pickle/i, '对象序列化'],
  [/tensorboard/i, '训练可视化'],
  [/pyplot|matplotlib/i, '绘图'],
  [/pythonic/i, '地道写法'],
  [/cold.?knowledge/i, '冷知识'],
  [/for[- ]?loop/i, '循环写法'],
  [/python.?basic/i, '基础语法'],
];

function hasCJK(text: string): boolean {
  return CJK.test(text);
}

function tidyGist(raw: string, max = 12): string {
  const cleaned = raw
    .replace(/[`*_~#>\[\]()|/\\]/g, '')
    .replace(/^\d+[\.\、]\s*/, '')
    .replace(/^(可以|将|把|是|在|用|对|与|和)+/, '')
    .replace(/\s+/g, '')
    .replace(/[。！？，、：:;]+$/g, '')
    .trim();

  if (!cleaned) return '';
  if (cleaned.length <= max) return cleaned;

  const cut = cleaned.slice(0, max);
  const pause = Math.max(cut.lastIndexOf('、'), cut.lastIndexOf('，'), cut.lastIndexOf('：'));
  return pause >= 4 ? cut.slice(0, pause) : cut;
}

function firstChineseRun(text: string): string {
  const match = text.match(/[\u4e00-\u9fff][^\n。！？]{3,24}/);
  return match ? tidyGist(match[0]) : '';
}

function stripCode(body: string): string {
  return body.replace(/```[\s\S]*?```/g, '\n').replace(/`[^`]+`/g, ' ');
}

export function extractNoteGist(title: string, description = '', body = ''): string {
  if (description && hasCJK(description)) return tidyGist(description, 14);

  for (const [pattern, gist] of TITLE_HINTS) {
    if (pattern.test(title)) return gist;
  }

  const plain = stripCode(body);
  const heading = plain.match(/^##\s+(.+)$/m);
  if (heading?.[1] && hasCJK(heading[1])) return tidyGist(heading[1]);

  const fromBody = firstChineseRun(plain.replace(/^#.*$/m, ''));
  if (fromBody) return fromBody;

  if (hasCJK(title)) return tidyGist(title, 10);
  return '';
}
