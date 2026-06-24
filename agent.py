"""
最小 Agent —— 不用任何框架，只用原生 Python + 标准库 HTTP。
能查天气、能算数学、能记住你的名字。核心逻辑不到 150 行。

它揭穿一个事实：所谓 Agent，内核就是一个循环——
让大模型自己决定"调哪个工具"，我们执行，把结果喂回去，直到它给出答案。
LangChain 帮你包的，正是这一圈。拆开看，它没那么神。

运行前：export DEEPSEEK_API_KEY=你的key
运行：    python3 agent.py
"""

import os
import re
import json
import urllib.request
import urllib.parse

LLM_URL = "https://api.deepseek.com/chat/completions"
LLM_MODEL = "deepseek-chat"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# ---------- 一次 HTTP 请求，就是和大模型说话的全部 ----------

def call_llm(messages):
    body = json.dumps({"model": LLM_MODEL, "messages": messages,
                       "temperature": 0}).encode("utf-8")
    req = urllib.request.Request(
        LLM_URL, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


# ---------- 三个工具：本质就是三个普通函数 ----------

MEMORY = {}  # Agent 的"记忆"，朴素到只是一个字典

def tool_weather(city):
    """查天气：wttr.in 免费、不要 key。"""
    url = "https://wttr.in/%s?format=3" % urllib.parse.quote(city)
    with urllib.request.urlopen(url, timeout=20) as resp:
        return resp.read().decode("utf-8").strip()

def tool_calc(expr):
    """算数学：只允许数字和运算符，防止 eval 被滥用。"""
    if not re.fullmatch(r"[0-9+\-*/().%\s]+", expr):
        return "错误：只支持基础四则运算"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return "计算出错：%s" % e

def tool_remember(text):
    """记名字：写进记忆字典。"""
    MEMORY["name"] = text.strip()
    return "好的，我记住了：%s" % MEMORY["name"]

def tool_recall(_=""):
    """回忆名字：从记忆字典里取。"""
    return MEMORY.get("name", "（我还不知道你的名字）")

TOOLS = {
    "weather": tool_weather,
    "calc": tool_calc,
    "remember_name": tool_remember,
    "recall_name": tool_recall,
}

# ---------- 系统提示词：教会模型"用 JSON 表达它想干嘛" ----------

SYSTEM = """你是一个能调用工具的助手。每一步，你只能输出一个 JSON，二选一：
1) 调用工具：{"tool": "工具名", "args": "参数"}
2) 给出最终答案：{"answer": "给用户的话"}

可用工具：
- weather(城市)：查实时天气
- calc(算式)：做四则运算，如 "3*(4+5)"
- remember_name(名字)：记住用户的名字
- recall_name()：回忆用户的名字

规则：需要外部信息或计算时，先调工具；拿到工具结果后，再决定继续调用还是给出答案。
只输出 JSON，不要多余的话。"""


def extract_json(text):
    """模型偶尔会带点多余字符，抠出第一个 JSON 对象。"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {"answer": text}


# ---------- Agent 的心脏：think → act → observe 循环 ----------

def run_agent(user_input, history):
    messages = [{"role": "system", "content": SYSTEM}] + history
    messages.append({"role": "user", "content": user_input})

    for _ in range(8):  # 最多循环 8 步，防止无限调用
        reply = call_llm(messages)
        step = extract_json(reply)

        if "answer" in step:                      # 模型决定收尾
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": step["answer"]})
            return step["answer"]

        name = step.get("tool")
        if name not in TOOLS:                      # 模型调了不存在的工具
            observation = "没有这个工具：%s" % name
        else:
            observation = TOOLS[name](step.get("args", ""))
            print("   [调用 %s(%s) → %s]" % (name, step.get("args", ""), observation))

        # 把"模型的决定"和"工具的结果"都喂回去，进入下一轮
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user",
                         "content": "工具结果：%s" % observation})

    return "（想太久了，没能给出答案）"


# ---------- 一个最简单的对话循环 ----------

def main():
    if not API_KEY:
        print("请先设置环境变量 DEEPSEEK_API_KEY")
        return
    print("最小 Agent 已启动（输入 q 退出）。试试：")
    print("  · 北京天气怎么样？")
    print("  · 帮我算 (128+72)*3")
    print("  · 我叫张三，记住我  /  我叫什么名字？\n")
    history = []
    while True:
        try:
            user = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("q", "quit", "exit"):
            break
        if not user:
            continue
        print("助手 >", run_agent(user, history), "\n")


if __name__ == "__main__":
    main()
