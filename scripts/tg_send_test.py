"""
用你的 Telegram 账号给 Bot 发测试消息。
运行后会要求输入验证码。
"""
import asyncio
from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.network import ConnectionTcpMTProxyRandomizedIntermediate

# 使用 Telegram Desktop 公开凭证
API_ID = 2040
API_HASH = "b18441a1ff607e10a989891a5462e627"
PHONE = "+573117540175"
BOT_USERNAME = "ksnzizjwns_bot"

TEST_MESSAGES = [
    "BYD海豹 CIF价格到拉各斯是多少",
    "有什么20000美金以内的SUV推荐吗",
    "吉利星越L和哈弗H6哪个更适合出口非洲",
    "帮我出一份比亚迪汉 10辆的报价合同 买家是Lagos Motors",
]

async def main():
    client = TelegramClient(
        StringSession(), API_ID, API_HASH,
        proxy=("socks5", "127.0.0.1", 7890),
        connection_retries=3,
        timeout=30,
    )
    await client.start(phone=PHONE)
    print("登录成功！")

    for i, msg in enumerate(TEST_MESSAGES, 1):
        print(f"发送第{i}条: {msg}")
        await client.send_message(BOT_USERNAME, msg)
        await asyncio.sleep(15)  # 等 Bot 回复

    session_str = client.session.save()
    # 保存 session 供下次使用
    with open("scripts/.tg_session", "w") as f:
        f.write(session_str)
    print("\n全部发完！session 已保存，下次无需再输验证码。")
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
