"""
两步登录：
  步骤1: python tg_login.py request   -> 请求验证码
  步骤2: python tg_login.py verify 12345 -> 填入验证码完成登录并发测试消息
"""
import asyncio, sys
from telethon import TelegramClient
from telethon.sessions import StringSession

API_ID = 2040
API_HASH = "b18441a1ff607e10a989891a5462e627"
PHONE = "+573117540175"
BOT_USERNAME = "ksnzizjwns_bot"
SESSION_FILE = "scripts/.tg_session"
PHONE_HASH_FILE = "scripts/.tg_phone_hash"

TEST_MESSAGES = [
    "BYD海豹 CIF价格到拉各斯是多少",
    "有什么20000美金以内的SUV推荐吗",
    "吉利星越L和哈弗H6哪个更适合出口非洲",
    "帮我出一份比亚迪汉 10辆的报价合同 买家是Lagos Motors",
]

PROXY = ("socks5", "127.0.0.1", 7890)


async def request_code():
    client = TelegramClient(StringSession(), API_ID, API_HASH, proxy=PROXY)
    await client.connect()
    result = await client.send_code_request(PHONE)
    print(f"验证码已发送，phone_hash={result.phone_code_hash}")
    # 保存 phone_hash 和 session 供第二步使用
    with open(PHONE_HASH_FILE, "w") as f:
        f.write(result.phone_code_hash)
    with open(SESSION_FILE, "w") as f:
        f.write(client.session.save())
    await client.disconnect()
    print("验证码已发到你的 Telegram，告诉我收到的数字。")


async def verify_and_send(code: str):
    session = open(SESSION_FILE).read().strip()
    phone_hash = open(PHONE_HASH_FILE).read().strip()

    client = TelegramClient(StringSession(session), API_ID, API_HASH, proxy=PROXY)
    await client.connect()
    await client.sign_in(phone=PHONE, code=code, phone_code_hash=phone_hash)
    print("登录成功！开始发测试消息...")

    for i, msg in enumerate(TEST_MESSAGES, 1):
        print(f"  发送第{i}条: {msg}")
        await client.send_message(BOT_USERNAME, msg)
        await asyncio.sleep(20)

    # 保存 session 供下次复用
    with open(SESSION_FILE, "w") as f:
        f.write(client.session.save())
    print("全部发完！")
    await client.disconnect()


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "request"
    if step == "request":
        asyncio.run(request_code())
    elif step == "verify":
        code = sys.argv[2]
        asyncio.run(verify_and_send(code))
