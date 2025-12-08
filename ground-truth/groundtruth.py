#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import re  # 新增：用于压缩空白


def normalize_patch(p: str) -> str:
    """
    清理 patch 文本中的多余空格和换行：
    - 去掉每行末尾的多余空白
    - 去掉开头/结尾的空行
    - 将连续空行压缩为一个
    """
    lines = p.splitlines()

    # 去掉每行末尾的空格/Tab
    lines = [ln.rstrip() for ln in lines]

    # 去掉开头空行
    while lines and not lines[0].strip():
        lines.pop(0)

    # 去掉结尾空行
    while lines and not lines[-1].strip():
        lines.pop()

    # 压缩中间连续空行
    normalized_lines = []
    last_blank = False
    for ln in lines:
        if not ln.strip():  # 空行
            if last_blank:
                continue      # 跳过多余的空行
            last_blank = True
            normalized_lines.append("")  # 保留一个空行
        else:
            last_blank = False
            normalized_lines.append(ln)

    return "\n".join(normalized_lines)


def extract_patches_from_text(text: str):
    """
    从一个完整 .patch 文件内容中提取多个 patch（按 hunk 切分）。
    每个 patch 对应一个 @@ 开头的 hunk，去掉 --- / +++ / @@ 等前缀行，
    并做空格/换行的清理。
    """
    patches = []
    current = None  # 当前正在累计的 patch 内容（list[str]）

    for line in text.splitlines():
        # diff 头
        if line.startswith("diff --git "):
            if current:
                patches.append("\n".join(current).rstrip())
                current = None
            continue

        # 旧文件路径
        if line.startswith("--- "):
            if current:
                patches.append("\n".join(current).rstrip())
                current = None
            continue

        # 新文件路径
        if line.startswith("+++ "):
            continue

        # hunk 头部：@@ -xxx,yyy +aaa,bbb @@
        if line.startswith("@@ "):
            if current:
                patches.append("\n".join(current).rstrip())
            current = []
            continue

        # 其他 diff 头部信息
        if current is None:
            if (line.startswith("index ")
                    or line.startswith("new file mode")
                    or line.startswith("deleted file mode")):
                continue
            # 头部其他内容忽略
            continue

        # hunk 的内容行：可能以 ' ', '+', '-' 开头等
        current.append(line)

    # 最后一个 hunk
    if current:
        patches.append("\n".join(current).rstrip())

    # 去掉空 patch，并做空格/换行的清理
    cleaned = []
    for p in patches:
        if not p.strip():
            continue
        cleaned.append(normalize_patch(p))

    return cleaned


def iter_patch_files(root_dir: str):
    """遍历目录下所有 .patch 文件"""
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.endswith(".patch"):
                yield os.path.join(root, name)


# ========= 新增：压缩空白的函数 =========

def compact_patch_text(text: str) -> str:
    """
    把补丁内容压缩成一行：
    - 所有空白字符（空格/Tab/换行等）压缩为单个空格
    - 去掉首尾空白

    用在 Humanwrite 和 patch 字段上，避免 JSON 里出现大量 \\n 和缩进。
    """
    if text is None:
        return None
    # \s+ 匹配任意数量的空白（包括回车），替换成单个空格
    return re.sub(r"\s+", " ", text).strip()


# ========= bug_name 提取 & 人工补丁读取 =========

def extract_bug_name_from_generated_file(path: str):
    """
    从机器 patch 文件名中提取 bug_name。

    例如:
        patch2-Chart-14-ACS.patch -> Chart-14

    策略：
    - 去掉扩展名后按 '-' 拆分
    - 找到第一个纯数字 token，它前一个 token 当作项目名
    - bug_name = "<项目名>-<数字>"
    - 如果没匹配到，返回 None
    """
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)  # patch2-Chart-14-ACS
    parts = name.split("-")

    project = None
    bug_id = None
    for i, part in enumerate(parts):
        if part.isdigit() and i > 0:
            project = parts[i - 1]
            bug_id = part
            break

    if project and bug_id:
        return f"{project}-{bug_id}"
    else:
        return None


def get_human_patch_content(human_root: str,
                            bug_name: str,
                            cache: dict):
    """
    根据 bug_name 找到对应的人工补丁文件，并返回其内容（已压缩空白）。

    约定：
        bug_name: 例如 "Chart-14"
        human_root: 例如 "D4J-Humanwrite"
        文件路径:  human_root / "Chart" / "14.src.patch"

    用 cache 缓存，避免重复读取。
    """
    if not human_root or not bug_name:
        return None

    if bug_name in cache:
        return cache[bug_name]

    try:
        project, bug_id = bug_name.split("-", 1)
    except ValueError:
        print(f"警告: 无法从 bug_name 提取 project/bug_id: {bug_name}")
        cache[bug_name] = None
        return None

    patch_path = os.path.join(human_root, project, f"{bug_id}.src.patch")

    if not os.path.isfile(patch_path):
        print(f"警告: 找不到人工补丁文件: {patch_path} (bug_name={bug_name})")
        cache[bug_name] = None
        return None

    with open(patch_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # 先按 hunk 提取并规范化
    patches = extract_patches_from_text(content)
    if patches:
        human_text = "\n\n".join(patches)
    else:
        human_text = content.strip()

    # 再压缩空白，变成一行的字符串
    human_text = compact_patch_text(human_text)

    cache[bug_name] = human_text
    return human_text


# ========= 子目录处理 =========

def process_subdir(base_dir: str,
                   subdir_name: str,
                   label: int,
                   out_f,
                   human_root: str = None,
                   human_cache: dict = None):
    """
    处理 base_dir/subdir_name 下的所有 .patch 文件，
    提取补丁并写入到 out_f，每条记录包含：
        {
            "Humanwrite": <同 bug 的人工补丁文本（已压缩为一行）或 None>,
            "patch": <当前机器补丁 hunk（已压缩为一行）>,
            "label": <0 or 1>,
            "bug_name": <如 Chart-14>
        }
    """
    sub_path = os.path.join(base_dir, subdir_name)
    if not os.path.isdir(sub_path):
        print(f"警告: 目录不存在，跳过：{sub_path}")
        return

    for patch_file in iter_patch_files(sub_path):
        with open(patch_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # 机器补丁按 hunk 提取 & 初步规范化
        patches = extract_patches_from_text(content)
        bug_name = extract_bug_name_from_generated_file(patch_file)

        if human_cache is None:
            human_cache = {}

        human_text = get_human_patch_content(human_root, bug_name, human_cache) \
            if (human_root and bug_name) else None

        for p in patches:
            # 把每个机器 hunk 压缩为空白为一行
            compact_p = compact_patch_text(p)

            record = {
                "Humanwrite": human_text,  # D4J-Humanwrite/... 的补丁内容（同一个 bug）
                "patch": compact_p,        # 当前机器补丁 hunk
                "label": label,            # correct -> 0, overfitting -> 1
                "bug_name": bug_name,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "从指定目录下的 correct/ 和 overfitting/ 中提取 .patch 补丁，"
            "correct 标记为 0，overfitting 标记为 1；"
            "同时根据 bug_name 在人工补丁目录中找到对应的人工补丁内容，"
            "输出 JSONL，每行为 {Humanwrite, patch, label, bug_name}。"
            "Humanwrite 和 patch 中的空白会被压缩成一行，以避免大量回车和空格。"
        )
    )
    parser.add_argument("root_dir", help="根目录（包含 correct 和 overfitting 子目录）")
    parser.add_argument("output", help="输出 JSONL 文件路径")
    parser.add_argument(
        "--human_root",
        help=(
            "人工书写补丁根目录，例如 D4J-Humanwrite，"
            "内部结构类似 D4J-Humanwrite/Chart/14.src.patch"
        ),
    )

    args = parser.parse_args()

    human_cache = {}

    with open(args.output, "w", encoding="utf-8") as out_f:
        # correct -> label 0
        process_subdir(
            base_dir=args.root_dir,
            subdir_name="correct",
            label=0,
            out_f=out_f,
            human_root=args.human_root,
            human_cache=human_cache,
        )
        # overfitting -> label 1
        process_subdir(
            base_dir=args.root_dir,
            subdir_name="overfitting",
            label=1,
            out_f=out_f,
            human_root=args.human_root,
            human_cache=human_cache,
        )


if __name__ == "__main__":
    main()
