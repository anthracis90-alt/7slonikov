import os
import re
import json
import random
import threading
import time
from typing import Optional, Dict, Any, Tuple

import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import uvicorn

from sqlalchemy import create_engine, select, Integer, String, Text, Boolean
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes


# ==========================
# CONFIG (ENV)
# ==========================
VK_GROUP_ID = int(os.getenv("VK_GROUP_ID", "227395470"))  # club227395470
VK_POST_ID = int(os.getenv("VK_POST_ID", "385"))          # wall-227395470_385
VK_ACCESS_TOKEN = os.getenv("VK_ACCESS_TOKEN", "").strip()

VK_CALLBACK_SECRET = os.getenv("VK_CALLBACK_SECRET", "secret7slonikov").strip()
VK_CONFIRMATION_STRING = os.getenv("VK_CONFIRMATION_STRING", "").strip()

CODEWORD = os.getenv("CODEWORD", "Крутить").strip()

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHANNEL = os.getenv("TG_CHANNEL", "@sevenslonikov").strip()
TG_BOT_LINK = os.getenv("TG_BOT_LINK", "https://t.me/sevenelephant_bot").strip().replace("@", "")

TOTAL_ATTEMPTS = int(os.getenv("TOTAL_ATTEMPTS", "100000"))
TG_BONUS_ATTEMPTS = int(os.getenv("TG_BONUS_ATTEMPTS", "3"))

# 1 попытка раз в сутки (24 часа)
DAILY_COOLDOWN_SECONDS = int(os.getenv("DAILY_COOLDOWN_SECONDS", "86400"))

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///raffle_7slonikov.sqlite3")

VK_API = "https://api.vk.com/method"
VK_VER = "5.199"
VK_ID_RE = re.compile(r"^\d+$")


def tg_channel_link() -> str:
    ch = TG_CHANNEL.strip()
    if ch.startswith("@"):
        ch = ch[1:]
    return f"https://t.me/{ch}"


# ==========================
# DB
# ==========================
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class Participant(Base):
    __tablename__ = "participants"

    vk_user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tg_user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    post_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    comment_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    comment_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    vk_member_ok: Mapped[bool] = mapped_column(Boolean, default=False)

    tg_member_ok: Mapped[bool] = mapped_column(Boolean, default=False)
    tg_bonus_granted: Mapped[bool] = mapped_column(Boolean, default=False)

    attempts_used: Mapped[int] = mapped_column(Integer, default=0)

    # Время последней попытки (epoch seconds)
    last_attempt_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    result_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    result_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class PrizeInventory(Base):
    __tablename__ = "prize_inventory"

    code: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    remaining: Mapped[int] = mapped_column(Integer)


class RaffleState(Base):
    __tablename__ = "raffle_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # always 1
    remaining_attempts: Mapped[int] = mapped_column(Integer)    # из TOTAL_ATTEMPTS
    remaining_prizes: Mapped[int] = mapped_column(Integer)      # всего 100


PRIZES = [
    ("SCHEME_20", "🎁 Схема на выбор (из 20)", 20),
    ("DISCOUNT_50", "🏷 Скидка 50% на схемы для вышивания", 80),
]


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        st = db.get(RaffleState, 1)
        if st is None:
            total_prizes = sum(x[2] for x in PRIZES)  # 100
            db.add(RaffleState(id=1, remaining_attempts=TOTAL_ATTEMPTS, remaining_prizes=total_prizes))

        for code, title, count in PRIZES:
            if db.get(PrizeInventory, code) is None:
                db.add(PrizeInventory(code=code, title=title, remaining=count))

        db.commit()


# ==========================
# VK helpers
# ==========================
def vk_api_call(method: str, params: Dict[str, Any], http_method: str = "GET") -> Dict[str, Any]:
    params = dict(params)
    params["v"] = VK_VER
    params["access_token"] = VK_ACCESS_TOKEN

    url = f"{VK_API}/{method}"
    try:
        if http_method.upper() == "POST":
            r = requests.post(url, data=params, timeout=15)
        else:
            r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "error" in data:
            print("VK API ERROR:", data["error"])
        return data
    except Exception as e:
        print("VK API EXCEPTION:", repr(e))
        return {"error": {"error_msg": str(e)}}


def vk_groups_is_member(user_id: int) -> bool:
    if not VK_ACCESS_TOKEN:
        print("VK_ACCESS_TOKEN пустой — не могу проверить подписку на ВК.")
        return False
    data = vk_api_call("groups.isMember", {"group_id": VK_GROUP_ID, "user_id": user_id}, "GET")
    if "error" in data:
        return False
    return bool(data.get("response", 0))


def vk_create_comment(post_id: int, message: str, reply_to_comment: Optional[int] = None) -> None:
    if not VK_ACCESS_TOKEN:
        print("VK_ACCESS_TOKEN пустой — не могу писать комментарии.")
        return
    params: Dict[str, Any] = {
        "owner_id": -VK_GROUP_ID,
        "post_id": post_id,
        "message": message,
    }
    if reply_to_comment is not None:
        params["reply_to_comment"] = reply_to_comment
    _ = vk_api_call("wall.createComment", params, "POST")


# ==========================
# Codeword
# ==========================
def comment_has_codeword(text: str) -> bool:
    t = (text or "").strip().lower()
    cw = (CODEWORD or "").strip().lower()
    if not cw:
        return False
    return cw in t


# ==========================
# Raffle algorithm (100 prizes / 100000 attempts)
# ==========================
def _pick_any_available_prize(db, rng: random.Random) -> Optional[PrizeInventory]:
    prizes = db.scalars(select(PrizeInventory).where(PrizeInventory.remaining > 0)).all()
    if not prizes:
        return None
    return rng.choice(prizes)


def draw_one_attempt(db, vk_user_id: int, rng: random.Random) -> Tuple[str, str]:
    p = db.get(Participant, vk_user_id)
    if p is None:
        return "NONE", "Заявка не найдена."

    if p.result_code is not None:
        return p.result_code, (p.result_text or "")

    st = db.get(RaffleState, 1)
    if st is None:
        return "NONE", "Состояние розыгрыша не найдено."

    if st.remaining_attempts <= 0:
        return "NONE", "Попытки розыгрыша закончились."
    if st.remaining_prizes <= 0:
        return "NONE", "Призы уже закончились."

    # тратим 1 попытку
    st.remaining_attempts -= 1
    p.attempts_used += 1
    p.last_attempt_at = int(time.time())

    # вероятность выигрыша “без возврата”: remaining_prizes / attempts_total
    win_prob = st.remaining_prizes / (st.remaining_attempts + 1)
    if rng.random() >= win_prob:
        db.commit()
        return "NONE", "Не повезло 😔 В этот раз приз не выпал."

    prize = _pick_any_available_prize(db, rng)
    if prize is None:
        db.commit()
        return "NONE", "Призы уже закончились."

    prize.remaining -= 1
    st.remaining_prizes -= 1

    p.result_code = prize.code
    p.result_text = f"🎉 Поздравляем! Вам выпал приз: {prize.title}"
    db.commit()
    return p.result_code, p.result_text


def format_remaining_time(seconds: int) -> str:
    if seconds < 0:
        seconds = 0
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"~{hours} ч {minutes} мин"


# ==========================
# FastAPI (VK Callback)
# ==========================
app = FastAPI()


@app.on_event("startup")
async def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/vk/callback")
async def vk_callback(req: Request):
    try:
        raw = await req.body()
        payload: Dict[str, Any] = json.loads(raw.decode("utf-8") or "{}")
    except Exception:
        return PlainTextResponse("ok", status_code=200)

    # confirmation от ВК
    if payload.get("type") == "confirmation":
        return PlainTextResponse(VK_CONFIRMATION_STRING or "NO_CONFIRMATION_STRING_SET", status_code=200)

    # secret check
    if VK_CALLBACK_SECRET and payload.get("secret") != VK_CALLBACK_SECRET:
        return PlainTextResponse("ok", status_code=200)

    # интересуют только новые комментарии
    if payload.get("type") != "wall_reply_new":
        return PlainTextResponse("ok", status_code=200)

    obj = payload.get("object") or {}
    post_id = int(obj.get("post_id", 0))
    comment_id = int(obj.get("id", 0))
    from_id = int(obj.get("from_id", 0))
    text = (obj.get("text") or "").strip()

    print("VK EVENT:", {"post_id": post_id, "comment_id": comment_id, "from_id": from_id, "text": text})

    if post_id != VK_POST_ID:
        return PlainTextResponse("ok", status_code=200)

    if not comment_has_codeword(text):
        return PlainTextResponse("ok", status_code=200)

    # обязательна подписка на ВК
    if not vk_groups_is_member(from_id):
        vk_create_comment(
            post_id=post_id,
            reply_to_comment=comment_id,
            message="Чтобы участвовать — подпишитесь на группу ВК ✅ и напишите кодовое слово ещё раз."
        )
        return PlainTextResponse("ok", status_code=200)

    with SessionLocal() as db:
        p = db.get(Participant, from_id)
        if p is None:
            p = Participant(vk_user_id=from_id)
            db.add(p)
            db.commit()

        # антидубликат (на случай повторной доставки)
        if p.comment_id == comment_id and p.post_id == post_id:
            return PlainTextResponse("ok", status_code=200)

        # 1 попытка раз в 24 часа (если бонусные TG не используются)
        now = int(time.time())
        if p.last_attempt_at is not None and (now - p.last_attempt_at) < DAILY_COOLDOWN_SECONDS:
            remaining = DAILY_COOLDOWN_SECONDS - (now - p.last_attempt_at)
            msg = (
                "⏳ Сегодня вы уже использовали попытку.\n\n"
                f"Следующая попытка будет доступна через {format_remaining_time(remaining)}.\n\n"
                f"Хотите +{TG_BONUS_ATTEMPTS} бонусные попытки? ✅\n"
                f"1) Подпишитесь на Telegram-канал: {tg_channel_link()}\n"
                f"2) Напишите боту:\n"
                f"/start {from_id}\n"
                f"Бот: {TG_BOT_LINK}"
            )
            vk_create_comment(post_id=post_id, reply_to_comment=comment_id, message=msg)
            return PlainTextResponse("ok", status_code=200)

        # сохраняем комментарий
        p.post_id = post_id
        p.comment_id = comment_id
        p.comment_text = text
        p.vk_member_ok = True
        db.commit()

        rng = random.Random()
        code, result = draw_one_attempt(db, from_id, rng)

    # ответ в ВК
    if code != "NONE":
        vk_create_comment(post_id=post_id, reply_to_comment=comment_id, message=result)
    else:
        vk_create_comment(
            post_id=post_id,
            reply_to_comment=comment_id,
            message=(
                f"{result}\n\n"
                f"Приходите завтра за новой попыткой! ⏰\n\n"
                f"Хотите +{TG_BONUS_ATTEMPTS} бонусные попытки? ✅\n"
                f"Подпишитесь на Telegram-канал: {tg_channel_link()}\n"
                f"И напишите боту:\n"
                f"/start {from_id}\n"
                f"Бот: {TG_BOT_LINK}"
            )
        )

    return PlainTextResponse("ok", status_code=200)


# ==========================
# Telegram bot (polling)
# ==========================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user is None or update.message is None:
        return

    args = context.args or []
    if not args or not VK_ID_RE.match(args[0]):
        await update.message.reply_text("Команда: /start <ваш_vk_id>\nПример: /start 123456")
        return

    vk_id = int(args[0])
    tg_id = update.effective_user.id

    # Проверка подписки на канал
    try:
        member = await context.bot.get_chat_member(chat_id=TG_CHANNEL, user_id=tg_id)
        status = getattr(member, "status", None)
        tg_ok = status in ("member", "administrator", "creator")
    except Exception as e:
        print("TG get_chat_member error:", repr(e))
        tg_ok = False

    with SessionLocal() as db:
        p = db.get(Participant, vk_id)
        if p is None:
            await update.message.reply_text("Сначала напишите кодовое слово в комментарии под постом ВК.")
            return

        if p.result_code is not None:
            await update.message.reply_text(p.result_text or "Вы уже получили результат.")
            return

        p.tg_user_id = tg_id
        p.tg_member_ok = tg_ok
        db.commit()

        if not tg_ok:
            await update.message.reply_text(
                "Чтобы получить бонусные попытки, подпишитесь на Telegram-канал ✅\n"
                f"{tg_channel_link()}\n\n"
                "После подписки повторите:\n"
                f"/start {vk_id}"
            )
            return

        if p.tg_bonus_granted:
            await update.message.reply_text("Бонусные попытки уже были выданы ранее ✅")
            return

        p.tg_bonus_granted = True
        db.commit()

        rng = random.Random()

        # даём +3 бонусные попытки сразу
        results = []
        for i in range(TG_BONUS_ATTEMPTS):
            code, res = draw_one_attempt(db, vk_id, rng)
            results.append(f"Попытка {i + 1}: {res}")
            if code != "NONE":
                break

    await update.message.reply_text("✅ Бонус активирован!\n\n" + "\n\n".join(results))


def run_telegram_polling():
    if not TG_BOT_TOKEN:
        print("TG_BOT_TOKEN пустой — Telegram бот не запущен.")
        return

    import asyncio

    async def _runner():
        application = Application.builder().token(TG_BOT_TOKEN).build()
        application.add_handler(CommandHandler("start", start_cmd))
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        print("Telegram polling started.")
        while True:
            await asyncio.sleep(3600)

    asyncio.run(_runner())


if __name__ == "__main__":
    print("Starting 7slonikov raffle bot...")
    print("Listening on http://127.0.0.1:8000")
    print("VK_GROUP_ID =", VK_GROUP_ID, "VK_POST_ID =", VK_POST_ID, "CODEWORD =", CODEWORD)
    print("TOTAL_ATTEMPTS =", TOTAL_ATTEMPTS, "DAILY_COOLDOWN_SECONDS =", DAILY_COOLDOWN_SECONDS)
    print("TG_CHANNEL =", TG_CHANNEL, "TG_BOT_LINK =", TG_BOT_LINK)
    print("VK_CALLBACK_SECRET =", VK_CALLBACK_SECRET)
    print("VK_CONFIRMATION_STRING =", VK_CONFIRMATION_STRING if VK_CONFIRMATION_STRING else "<EMPTY>")
    print("TG_BOT_TOKEN =", "<SET>" if TG_BOT_TOKEN else "<EMPTY>")

    t = threading.Thread(target=run_telegram_polling, daemon=True)
    t.start()

    if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000"))
    )


