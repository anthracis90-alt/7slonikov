import os
import re
import json
import random
import threading
import time
import io
import csv
import asyncio
from typing import Optional, Dict, Any, Tuple, List

import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn

from sqlalchemy import create_engine, select, Integer, String, Text, Boolean, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column

from telegram import Update, Bot, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes


# ==========================
# CONFIG (ENV)
# ==========================
VK_GROUP_ID = int(os.getenv("VK_GROUP_ID", "227395470"))
VK_POST_ID = int(os.getenv("VK_POST_ID", "727"))
VK_ACCESS_TOKEN = os.getenv("VK_ACCESS_TOKEN", "").strip()

VK_CALLBACK_SECRET = os.getenv("VK_CALLBACK_SECRET", "secret7slonikov").strip()
VK_CONFIRMATION_STRING = os.getenv("VK_CONFIRMATION_STRING", "").strip()

CODEWORD = os.getenv("CODEWORD", "Крутить").strip()

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHANNEL = os.getenv("TG_CHANNEL", "@sevenslonikov").strip()
TG_BOT_LINK = os.getenv("TG_BOT_LINK", "https://t.me/sevenelephant_bot").strip().replace("@", "")

TOTAL_ATTEMPTS = int(os.getenv("TOTAL_ATTEMPTS", "100000"))
TG_BONUS_ATTEMPTS = int(os.getenv("TG_BONUS_ATTEMPTS", "3"))
DAILY_COOLDOWN_SECONDS = int(os.getenv("DAILY_COOLDOWN_SECONDS", "86400"))

# шанс выигрыша (0.05 = ~каждый 20-й)
WIN_PROB = float(os.getenv("WIN_PROB", "0.05"))

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///raffle_7slonikov.sqlite3")

VK_API = "https://api.vk.com/method"
VK_VER = "5.199"
VK_ID_RE = re.compile(r"^\d+$")

# Webhook settings
RUN_MODE = os.getenv("RUN_MODE", "").strip().lower()  # webhook / polling
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip()

# Admin IDs
ADMIN_TG_IDS_RAW = os.getenv("ADMIN_TG_IDS", "").strip()
ADMIN_TG_IDS = {int(x) for x in re.split(r"[,\s]+", ADMIN_TG_IDS_RAW) if x.isdigit()}
ADMIN_WINNERS_MAX = int(os.getenv("ADMIN_WINNERS_MAX", "200"))

# Auto-backup (seconds). Default: 24h
AUTO_BACKUP_INTERVAL_SECONDS = int(os.getenv("AUTO_BACKUP_INTERVAL_SECONDS", "86400"))

# Optional: force reset prize inventory on startup (dangerous for running raffle)
RESET_PRIZES = os.getenv("RESET_PRIZES", "0").strip() == "1"


def tg_channel_link() -> str:
    ch = TG_CHANNEL.strip()
    if ch.startswith("@"):
        ch = ch[1:]
    return f"https://t.me/{ch}"


def now_ts() -> int:
    return int(time.time())


def fmt_wait(seconds: int) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"~{h} ч {m} мин"


def pretty_ratio(prob: float) -> str:
    try:
        prob = float(prob)
        if prob <= 0:
            return "∞"
        return str(int(round(1.0 / prob)))
    except Exception:
        return "?"


def sanitize_db_url(url: str) -> str:
    """
    Hide password in URL: postgresql+psycopg://user:pass@host/db -> postgresql+psycopg://user:***@host/db
    """
    if not url:
        return "<EMPTY>"
    # replace user:password@
    return re.sub(r":([^:@/]+)@", r":***@", url)


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

    bonus_remaining: Mapped[int] = mapped_column(Integer, default=0)

    attempts_used: Mapped[int] = mapped_column(Integer, default=0)
    last_daily_attempt_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

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
    remaining_attempts: Mapped[int] = mapped_column(Integer)
    remaining_prizes: Mapped[int] = mapped_column(Integer)


class Winner(Base):
    __tablename__ = "winners"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[int] = mapped_column(Integer)
    vk_user_id: Mapped[int] = mapped_column(Integer)
    prize_code: Mapped[str] = mapped_column(String(64))
    prize_title: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(32))  # vk_daily / vk_bonus
    post_id: Mapped[int] = mapped_column(Integer)
    comment_id: Mapped[int] = mapped_column(Integer)


# ✅ NEW PRIZES: 10 + 70 = 80
PRIZES = [
    ("SCHEME_10", "🎁 Схема на выбор (из 10)", 10),
    ("DISCOUNT_50", "🏷 Скидка 50% на схемы для вышивания", 70),
]


def _sqlite_add_column_if_missing(conn, table: str, column: str, coltype_sql: str, default_sql: Optional[str] = None):
    cols = conn.execute(text(f"PRAGMA table_info({table});")).fetchall()
    col_names = {c[1] for c in cols}
    if column in col_names:
        return
    sql = f"ALTER TABLE {table} ADD COLUMN {column} {coltype_sql}"
    if default_sql is not None:
        sql += f" DEFAULT {default_sql}"
    conn.execute(text(sql))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)

    # Soft migration for old sqlite DB
    if DATABASE_URL.startswith("sqlite"):
        with engine.begin() as conn:
            try:
                _sqlite_add_column_if_missing(conn, "participants", "bonus_remaining", "INTEGER", "0")
                _sqlite_add_column_if_missing(conn, "participants", "last_daily_attempt_at", "INTEGER", None)
                _sqlite_add_column_if_missing(conn, "participants", "attempts_used", "INTEGER", "0")
                _sqlite_add_column_if_missing(conn, "participants", "tg_bonus_granted", "BOOLEAN", "0")
                _sqlite_add_column_if_missing(conn, "participants", "tg_member_ok", "BOOLEAN", "0")
                _sqlite_add_column_if_missing(conn, "participants", "vk_member_ok", "BOOLEAN", "0")
            except Exception as e:
                print("SQLite migration error:", repr(e))

    with SessionLocal() as db:
        st = db.get(RaffleState, 1)
        if st is None:
            total_prizes = sum(x[2] for x in PRIZES)
            db.add(RaffleState(id=1, remaining_attempts=TOTAL_ATTEMPTS, remaining_prizes=total_prizes))
            db.commit()

        # Ensure inventory rows exist; if already exist — cap to new counts
        if RESET_PRIZES:
            # full reset (dangerous)
            db.execute(text("DELETE FROM prize_inventory"))
            db.commit()

        for code, title, count in PRIZES:
            row = db.get(PrizeInventory, code)
            if row is None:
                db.add(PrizeInventory(code=code, title=title, remaining=count))
            else:
                row.title = title
                # If lowering prize counts, cap remaining
                if row.remaining > count:
                    row.remaining = count

        db.commit()

        # Recompute remaining_prizes from inventory (safer after edits)
        inv = db.scalars(select(PrizeInventory)).all()
        total_remaining = sum(x.remaining for x in inv)
        st = db.get(RaffleState, 1)
        if st:
            st.remaining_prizes = total_remaining
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
def comment_has_codeword(text_val: str) -> bool:
    t = (text_val or "").strip().lower()
    cw = (CODEWORD or "").strip().lower()
    if not cw:
        return False
    return cw in t


# ==========================
# Prize / Winner record
# ==========================
def _pick_any_available_prize(db, rng: random.Random) -> Optional[PrizeInventory]:
    prizes = db.scalars(select(PrizeInventory).where(PrizeInventory.remaining > 0)).all()
    if not prizes:
        return None
    return rng.choice(prizes)


def _record_winner(db, vk_user_id: int, prize_code: str, prize_title: str, source: str, post_id: int, comment_id: int):
    db.add(Winner(
        ts=now_ts(),
        vk_user_id=vk_user_id,
        prize_code=prize_code,
        prize_title=prize_title,
        source=source,
        post_id=post_id,
        comment_id=comment_id,
    ))


# ==========================
# Raffle
# ==========================
def draw_one_attempt(
    db,
    vk_user_id: int,
    rng: random.Random,
    *,
    source: str,
    comment_id: int,
    post_id: int
) -> Tuple[str, str]:
    p = db.get(Participant, vk_user_id)
    if p is None:
        return "NONE", "Заявка не найдена."

    if p.result_code is not None:
        return p.result_code, (p.result_text or "Вы уже получили приз.")

    st = db.get(RaffleState, 1)
    if st is None:
        return "NONE", "Состояние розыгрыша не найдено."

    if st.remaining_attempts <= 0:
        return "NONE", "Попытки розыгрыша закончились."
    if st.remaining_prizes <= 0:
        return "NONE", "Призы уже закончились."

    # consume global attempt
    st.remaining_attempts -= 1
    p.attempts_used += 1

    win_prob = max(0.0, min(float(WIN_PROB), 1.0))
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

    _record_winner(db, vk_user_id, prize.code, prize.title, source, post_id, comment_id)

    db.commit()
    return p.result_code, p.result_text


# ==========================
# Admin helpers
# ==========================
def is_admin(tg_user_id: int) -> bool:
    return tg_user_id in ADMIN_TG_IDS


def build_stats_text_plain() -> str:
    with SessionLocal() as db:
        st = db.get(RaffleState, 1)
        if not st:
            return "❌ raffle_state не найден"

        prizes = db.scalars(select(PrizeInventory)).all()
        prizes_sorted = sorted(prizes, key=lambda x: x.code)

        winners_total = db.scalar(select(text("COUNT(1)")).select_from(text("winners"))) or 0

        lines = []
        lines.append("📊 7 Слоников — статистика розыгрыша")
        lines.append("")
        lines.append(f"🎯 Осталось попыток: {st.remaining_attempts}")
        lines.append(f"🎁 Осталось призов: {st.remaining_prizes}")
        lines.append(f"🏆 Победителей (в БД): {winners_total}")
        lines.append("")
        lines.append("🎁 Остатки по призам:")
        for pr in prizes_sorted:
            lines.append(f"• {pr.title} — {pr.remaining}")
        lines.append("")
        lines.append(f"🎲 WIN_PROB = {WIN_PROB} (примерно 1 из {pretty_ratio(WIN_PROB)})")
        lines.append(f"🗄 DATABASE_URL = {sanitize_db_url(DATABASE_URL)}")
        return "\n".join(lines)


def build_db_info_text() -> str:
    from urllib.parse import urlparse
    u = DATABASE_URL or ""
    safe = sanitize_db_url(u)
    try:
        p = urlparse(u.replace("postgresql+psycopg", "postgresql"))
        host = p.hostname or ""
        port = p.port or ""
        dbname = (p.path or "").lstrip("/")
        scheme = p.scheme
    except Exception:
        host = port = dbname = scheme = "?"
    lines = [
        "🧩 DB INFO",
        f"• URL (safe): {safe}",
        f"• scheme: {scheme}",
        f"• host: {host}",
        f"• port: {port}",
        f"• db: {dbname}",
    ]
    return "\n".join(lines)


def build_winners_csv_bytes() -> Tuple[bytes, int]:
    with SessionLocal() as db:
        rows: List[Winner] = db.scalars(select(Winner).order_by(Winner.id.asc())).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "timestamp", "vk_user_id", "prize_code", "prize_title", "source", "post_id", "comment_id"])
    for w in rows:
        writer.writerow([w.id, w.ts, w.vk_user_id, w.prize_code, w.prize_title, w.source, w.post_id, w.comment_id])

    return output.getvalue().encode("utf-8"), len(rows)


# ==========================
# Telegram commands
# ==========================
async def admin_stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user is None or update.message is None:
        return
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Нет доступа.")
        return
    await update.message.reply_text(build_stats_text_plain())


async def admin_db_info_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user is None or update.message is None:
        return
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Нет доступа.")
        return
    await update.message.reply_text(build_db_info_text())


async def admin_winners_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user is None or update.message is None:
        return
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Нет доступа.")
        return

    n = 20
    if context.args and context.args[0].isdigit():
        n = int(context.args[0])
    n = max(1, min(n, ADMIN_WINNERS_MAX))

    with SessionLocal() as db:
        rows: List[Winner] = db.scalars(
            select(Winner).order_by(Winner.id.desc()).limit(n)
        ).all()

    if not rows:
        await update.message.reply_text("Победителей пока нет.")
        return

    lines = [f"🏆 Последние победители (последние {len(rows)}):", ""]
    for w in rows:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(w.ts))
        lines.append(f"• {t} | vk_id={w.vk_user_id} | {w.prize_title} | {w.source}")
    await update.message.reply_text("\n".join(lines))


async def admin_export_csv_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user is None or update.message is None:
        return
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("⛔ Нет доступа.")
        return

    data_bytes, count_rows = build_winners_csv_bytes()
    bio = io.BytesIO(data_bytes)
    bio.name = "winners_export.csv"
    bio.seek(0)

    await update.message.reply_document(
        document=InputFile(bio, filename="winners_export.csv"),
        caption=f"Экспорт победителей: {count_rows} строк"
    )


# ==========================
# Telegram — bonus only
# ==========================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user is None or update.message is None:
        return

    args = context.args or []
    if not args or not VK_ID_RE.match(args[0]):
        await update.message.reply_text(
            "Команда: /start <ваш_vk_id>\n"
            "Пример: /start 123456\n\n"
            "После активации бонуса крутите в ВК: пишите комментарий с кодовым словом."
        )
        return

    vk_id = int(args[0])
    tg_id = update.effective_user.id

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
            await update.message.reply_text(
                "Я не вижу вашу заявку в ВК.\n"
                "Сначала напишите кодовое слово в комментарии под постом ВК."
            )
            return

        if p.result_code is not None:
            await update.message.reply_text(p.result_text or "Вы уже получили приз.")
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
            await update.message.reply_text(
                f"Бонус уже активирован ранее ✅\n"
                f"Осталось бонусных попыток: {p.bonus_remaining}\n\n"
                "Используйте попытки в ВК: пишите комментарий с кодовым словом под постом."
            )
            return

        p.tg_bonus_granted = True
        p.bonus_remaining = TG_BONUS_ATTEMPTS
        db.commit()

    await update.message.reply_text(
        "✅ Бонус активирован!\n\n"
        f"Вам добавлено +{TG_BONUS_ATTEMPTS} бонусные попытки.\n"
        "Используются по одной — только в ВК.\n\n"
        "👉 Вернитесь в ВК и пишите комментарий с кодовым словом под постом, чтобы крутить."
    )


# ==========================
# Auto-backup task (send CSV daily to admins)
# ==========================
async def auto_backup_loop(bot: Bot):
    if not ADMIN_TG_IDS:
        print("Auto-backup: ADMIN_TG_IDS empty -> disabled")
        return

    print(f"Auto-backup: enabled every {AUTO_BACKUP_INTERVAL_SECONDS}s for admins: {sorted(list(ADMIN_TG_IDS))}")
    while True:
        try:
            data_bytes, count_rows = build_winners_csv_bytes()
            bio = io.BytesIO(data_bytes)
            bio.name = "winners_backup.csv"
            bio.seek(0)

            caption = f"🗂 Авто-бэкап победителей (CSV)\nСтрок: {count_rows}\nUTC: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(now_ts()))}"

            for admin_id in list(ADMIN_TG_IDS):
                try:
                    bio.seek(0)
                    await bot.send_document(chat_id=admin_id, document=InputFile(bio, filename="winners_backup.csv"), caption=caption)
                except Exception as e:
                    print("Auto-backup send error to", admin_id, ":", repr(e))
        except Exception as e:
            print("Auto-backup loop error:", repr(e))

        await asyncio.sleep(max(60, AUTO_BACKUP_INTERVAL_SECONDS))


# ==========================
# FastAPI (VK + TG webhook)
# ==========================
app = FastAPI()
tg_app: Optional[Application] = None
_backup_task: Optional[asyncio.Task] = None


@app.on_event("startup")
async def on_startup():
    global tg_app, _backup_task
    init_db()

    if TG_BOT_TOKEN:
        tg_app = Application.builder().token(TG_BOT_TOKEN).build()
        tg_app.add_handler(CommandHandler("start", start_cmd))
        tg_app.add_handler(CommandHandler("admin_stats", admin_stats_cmd))
        tg_app.add_handler(CommandHandler("admin_winners", admin_winners_cmd))
        tg_app.add_handler(CommandHandler("admin_export_csv", admin_export_csv_cmd))
        tg_app.add_handler(CommandHandler("admin_db_info", admin_db_info_cmd))

        use_webhook = (RUN_MODE == "webhook") or bool(RENDER_EXTERNAL_URL)
        if use_webhook:
            await tg_app.initialize()
            await tg_app.start()

            base = RENDER_EXTERNAL_URL.rstrip("/")
            if not base:
                print("WARN: RENDER_EXTERNAL_URL empty; webhook not set.")
            else:
                webhook_url = f"{base}/tg/webhook"
                try:
                    bot = Bot(token=TG_BOT_TOKEN)
                    # IMPORTANT: drop pending updates to avoid conflicts after redeploy
                    await bot.delete_webhook(drop_pending_updates=True)
                    await bot.set_webhook(webhook_url)
                    print("Telegram webhook set:", webhook_url)
                except Exception as e:
                    print("Telegram webhook setup error:", repr(e))

            # start daily auto-backup loop (works in webhook mode)
            try:
                bot = Bot(token=TG_BOT_TOKEN)
                if _backup_task is None:
                    _backup_task = asyncio.create_task(auto_backup_loop(bot))
            except Exception as e:
                print("Auto-backup start error:", repr(e))


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

    # VK confirmation
    if payload.get("type") == "confirmation":
        return PlainTextResponse(VK_CONFIRMATION_STRING or "NO_CONFIRMATION_STRING_SET", status_code=200)

    # secret check
    if VK_CALLBACK_SECRET and payload.get("secret") != VK_CALLBACK_SECRET:
        return PlainTextResponse("ok", status_code=200)

    if payload.get("type") != "wall_reply_new":
        return PlainTextResponse("ok", status_code=200)

    obj = payload.get("object") or {}
    post_id = int(obj.get("post_id", 0))
    comment_id = int(obj.get("id", 0))
    from_id = int(obj.get("from_id", 0))
    text_val = (obj.get("text") or "").strip()

    print("VK EVENT:", {"post_id": post_id, "comment_id": comment_id, "from_id": from_id, "text": text_val})

    if post_id != VK_POST_ID:
        return PlainTextResponse("ok", status_code=200)

    if not comment_has_codeword(text_val):
        return PlainTextResponse("ok", status_code=200)

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

        if p.result_code is not None:
            vk_create_comment(post_id=post_id, reply_to_comment=comment_id, message=p.result_text or "Вы уже выиграли.")
            return PlainTextResponse("ok", status_code=200)

        # anti-duplicate
        if p.comment_id == comment_id and p.post_id == post_id:
            return PlainTextResponse("ok", status_code=200)

        p.post_id = post_id
        p.comment_id = comment_id
        p.comment_text = text_val
        p.vk_member_ok = True
        db.commit()

        now = now_ts()
        daily_available = (p.last_daily_attempt_at is None) or ((now - p.last_daily_attempt_at) >= DAILY_COOLDOWN_SECONDS)

        if daily_available:
            p.last_daily_attempt_at = now
            db.commit()
            rng = random.Random()
            code, result = draw_one_attempt(db, from_id, rng, source="vk_daily", comment_id=comment_id, post_id=post_id)
        elif p.bonus_remaining > 0:
            p.bonus_remaining -= 1
            db.commit()
            rng = random.Random()
            code, result = draw_one_attempt(db, from_id, rng, source="vk_bonus", comment_id=comment_id, post_id=post_id)
        else:
            remaining = DAILY_COOLDOWN_SECONDS - (now - (p.last_daily_attempt_at or now))
            msg = (
                "⏳ Сегодня дневная попытка уже использована.\n"
                f"Следующая дневная попытка будет через {fmt_wait(remaining)}.\n\n"
                f"Хотите +{TG_BONUS_ATTEMPTS} бонусные попытки? ✅\n"
                f"1) Подпишитесь на Telegram-канал: {tg_channel_link()}\n"
                f"2) Напишите боту:\n"
                f"/start {from_id}\n"
                f"Бот: {TG_BOT_LINK}"
            )
            vk_create_comment(post_id=post_id, reply_to_comment=comment_id, message=msg)
            return PlainTextResponse("ok", status_code=200)

    if code != "NONE":
        vk_create_comment(post_id=post_id, reply_to_comment=comment_id, message=result)
    else:
        vk_create_comment(
            post_id=post_id,
            reply_to_comment=comment_id,
            message=(
                f"{result}\n\n"
                "💡 Можно попробовать ещё раз:\n"
                "— следующая дневная попытка завтра\n"
                f"— или получите +{TG_BONUS_ATTEMPTS} бонусные попытки через Telegram ✅\n"
                f"{tg_channel_link()}\n"
                f"Бот: {TG_BOT_LINK}\n"
                f"Команда: /start {from_id}"
            )
        )

    return PlainTextResponse("ok", status_code=200)


@app.post("/tg/webhook")
async def tg_webhook(req: Request):
    if not TG_BOT_TOKEN or tg_app is None:
        return JSONResponse({"ok": False, "error": "telegram not configured"}, status_code=200)

    data = await req.json()
    upd = Update.de_json(data, tg_app.bot)
    try:
        await tg_app.process_update(upd)
    except Exception as e:
        print("tg_webhook process_update error:", repr(e))
    return JSONResponse({"ok": True}, status_code=200)


# ==========================
# Local polling (ONLY if no webhook)
# ==========================
def run_telegram_polling():
    if not TG_BOT_TOKEN:
        print("TG_BOT_TOKEN пустой — Telegram бот не запущен.")
        return

    import asyncio

    async def _runner():
        application = Application.builder().token(TG_BOT_TOKEN).build()
        application.add_handler(CommandHandler("start", start_cmd))
        application.add_handler(CommandHandler("admin_stats", admin_stats_cmd))
        application.add_handler(CommandHandler("admin_winners", admin_winners_cmd))
        application.add_handler(CommandHandler("admin_export_csv", admin_export_csv_cmd))
        application.add_handler(CommandHandler("admin_db_info", admin_db_info_cmd))
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        print("Telegram polling started.")

        # auto-backup in polling mode too
        bot = Bot(token=TG_BOT_TOKEN)
        asyncio.create_task(auto_backup_loop(bot))

        while True:
            await asyncio.sleep(3600)

    asyncio.run(_runner())


# ==========================
# Entrypoint
# ==========================
if __name__ == "__main__":
    print("Starting raffle bot...")
    print("VK_GROUP_ID =", VK_GROUP_ID, "VK_POST_ID =", VK_POST_ID, "CODEWORD =", CODEWORD)
    print("TOTAL_ATTEMPTS =", TOTAL_ATTEMPTS, "WIN_PROB =", WIN_PROB)
    print("DAILY_COOLDOWN_SECONDS =", DAILY_COOLDOWN_SECONDS, "TG_BONUS_ATTEMPTS =", TG_BONUS_ATTEMPTS)
    print("TG_CHANNEL =", TG_CHANNEL, "TG_BOT_LINK =", TG_BOT_LINK)
    print("RUN_MODE =", RUN_MODE, "RENDER_EXTERNAL_URL =", RENDER_EXTERNAL_URL)
    print("ADMIN_TG_IDS =", sorted(list(ADMIN_TG_IDS)) if ADMIN_TG_IDS else "<EMPTY>")
    print("DATABASE_URL =", sanitize_db_url(DATABASE_URL))
    print("PRIZES =", PRIZES)

    use_webhook = (RUN_MODE == "webhook") or bool(RENDER_EXTERNAL_URL)
    if not use_webhook:
        t = threading.Thread(target=run_telegram_polling, daemon=True)
        t.start()

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))





