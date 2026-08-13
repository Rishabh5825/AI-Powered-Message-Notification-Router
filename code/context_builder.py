"""
Stage 2b — Context Builder

Builds structured context for each message from the relational CSV data.
This is the "structured filter" half of the hybrid retrieval system —
it pulls all hard facts (user preferences, group membership, business
verification, notification fatigue) without needing embeddings.
"""

from dataclasses import dataclass, field
from datetime import datetime

from data_loader import DataStore


# ──────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────

@dataclass
class UserContext:
    """Aggregated user behavioral profile."""
    user_id: str = ""
    dnd_window: str = ""
    opened_30d: int = 0
    replied_30d: int = 0
    dismissed_30d: int = 0
    reported_30d: int = 0
    engagement_ratio: float = 0.0       # opened / (opened + dismissed)
    fatigue_ratio: float = 0.0          # avg daily dismissals / avg daily notifications
    avg_daily_notifications: float = 0.0
    avg_daily_dismissals: float = 0.0
    muted_groups: set = field(default_factory=set)
    admin_groups: set = field(default_factory=set)


@dataclass
class GroupContext:
    """Group-level context for a message."""
    group_id: str = ""
    group_name: str = ""
    group_type: str = ""
    member_count: int = 0
    sender_is_admin: bool = False
    user_has_muted: bool = False
    user_role: str = ""
    user_read_rate: float = 0.0
    user_dismissals_30d: int = 0
    user_messages_sent_30d: int = 0


@dataclass
class BusinessContext:
    """Business sender context."""
    business_id: str = ""
    display_name: str = ""
    brand_name: str = ""
    category: str = ""
    verified: bool = False
    official_domain: str = ""
    domain_used: str = ""
    domain_age_days: int = 999
    account_age_days: int = 999
    user_reports_30d: int = 0
    risk_score: float = 0.0
    # User-business relationship
    user_relationship: str = ""
    user_allows_promotions: bool = True
    user_opted_out: bool = False
    user_opens_30d: int = 0
    user_dismissals_30d: int = 0


@dataclass
class MessageContext:
    """Full assembled context for a single incoming message."""
    # Message fields
    message_id: str = ""
    user_id: str = ""
    conversation_type: str = ""
    message_text: str = ""
    media_type: str = ""
    media_content: str = ""             # Unified text from multimodal normalizer
    forwarded_count: int = 0
    created_at: str = ""
    sender_user_id: str = ""
    group_id: str = ""
    business_id: str = ""
    media_id: str = ""

    # Assembled context
    user_ctx: UserContext = field(default_factory=UserContext)
    group_ctx: GroupContext | None = None
    business_ctx: BusinessContext | None = None
    is_in_dnd: bool = False
    sender_exists: bool = False


# ──────────────────────────────────────────────────────────────────
# Builder Functions
# ──────────────────────────────────────────────────────────────────

def build_user_context(user_id: str, store: DataStore) -> UserContext:
    """Build an aggregated user profile from users.csv + daily_summary + group_members."""
    user = store.get_user(user_id)

    ctx = UserContext(user_id=user_id)
    ctx.dnd_window = user.get("do_not_disturb_window", "")
    ctx.opened_30d = int(user.get("messages_opened_30d", 0))
    ctx.replied_30d = int(user.get("messages_replied_30d", 0))
    ctx.dismissed_30d = int(user.get("notifications_dismissed_30d", 0))
    ctx.reported_30d = int(user.get("messages_reported_30d", 0))

    # Engagement ratio
    total = ctx.opened_30d + ctx.dismissed_30d
    ctx.engagement_ratio = ctx.opened_30d / max(total, 1)

    # Notification fatigue from daily_notification_summary
    daily = store.daily_summary.get(user_id, [])
    if daily:
        ctx.avg_daily_notifications = sum(
            int(d.get("notifications_sent", 0)) for d in daily
        ) / len(daily)
        ctx.avg_daily_dismissals = sum(
            int(d.get("notifications_dismissed", 0)) for d in daily
        ) / len(daily)
        ctx.fatigue_ratio = ctx.avg_daily_dismissals / max(
            ctx.avg_daily_notifications, 1
        )

    # Muted and admin groups
    memberships = store.group_members_by_user.get(user_id, [])
    ctx.muted_groups = {
        m["group_id"] for m in memberships if m.get("group_muted_by_user") == "1"
    }
    ctx.admin_groups = {
        m["group_id"] for m in memberships if m.get("role") == "admin"
    }

    return ctx


def build_group_context(
    group_id: str, user_id: str, sender_user_id: str, store: DataStore
) -> GroupContext:
    """Build group-level context for a group message."""
    group = store.get_group(group_id)
    membership = store.get_user_group_membership(user_id, group_id)
    sender_membership = store.get_user_group_membership(sender_user_id, group_id)

    ctx = GroupContext(group_id=group_id)
    ctx.group_name = group.get("group_name", "")
    ctx.group_type = group.get("group_type", "")
    ctx.member_count = int(group.get("member_count", 0))
    ctx.sender_is_admin = (
        sender_membership.get("role") == "admin" if sender_membership else False
    )

    if membership:
        ctx.user_has_muted = membership.get("group_muted_by_user") == "1"
        ctx.user_role = membership.get("role", "member")
        ctx.user_dismissals_30d = int(
            membership.get("notifications_dismissed_30d", 0)
        )
        ctx.user_messages_sent_30d = int(
            membership.get("messages_sent_30d", 0)
        )
        read = int(membership.get("messages_read_30d", 0))
        sent_in_group = int(
            store.get_group(group_id).get("messages_30d", 0)
        )
        ctx.user_read_rate = read / max(sent_in_group, 1)

    return ctx


def build_business_context(
    business_id: str, user_id: str, store: DataStore
) -> BusinessContext:
    """Build business sender context with risk scoring."""
    biz = store.get_business(business_id)
    relation = store.get_user_business_relation(user_id, business_id)

    ctx = BusinessContext(business_id=business_id)
    ctx.display_name = biz.get("display_name", "")
    ctx.brand_name = biz.get("brand_name", "")
    ctx.category = biz.get("category", "")
    ctx.verified = biz.get("verified") == "1"
    ctx.official_domain = biz.get("official_domain", "")
    ctx.domain_used = biz.get("domain_used_by_sender", "")
    ctx.domain_age_days = int(biz.get("domain_used_by_sender_age_days", 999))
    ctx.account_age_days = int(biz.get("account_age_days", 999))
    ctx.user_reports_30d = int(biz.get("user_reports_30d", 0))

    # Compute deterministic risk score
    ctx.risk_score = _compute_business_risk(biz)

    # User-business relationship
    if relation:
        ctx.user_relationship = relation.get("why_user_knows_account", "")
        ctx.user_allows_promotions = relation.get("allows_promotions") != "0"
        ctx.user_opted_out = bool(relation.get("promotions_opted_out_at", ""))
        ctx.user_opens_30d = int(relation.get("messages_opened_30d", 0))
        ctx.user_dismissals_30d = int(relation.get("messages_dismissed_30d", 0))

    return ctx


def _compute_business_risk(biz: dict) -> float:
    """
    Deterministic business risk score (0-100).
    Above 50 = likely scam/spam.

    Signals:
      - Unverified account:        +30
      - Domain mismatch:           +40
      - Fresh domain (<30 days):   +20
      - Many reports (>10):        +25
      - New account (<60 days):    +15
    """
    risk = 0
    if biz.get("verified") != "1":
        risk += 30
    if (
        biz.get("official_domain", "")
        and biz.get("domain_used_by_sender", "")
        and biz.get("official_domain") != biz.get("domain_used_by_sender")
    ):
        risk += 40
    if int(biz.get("domain_used_by_sender_age_days", 999)) < 30:
        risk += 20
    if int(biz.get("user_reports_30d", 0)) > 10:
        risk += 25
    if int(biz.get("account_age_days", 999)) < 60:
        risk += 15
    return min(risk, 100)


def is_in_dnd(msg_time: str, dnd_window: str) -> bool:
    """Check if a message timestamp falls within the user's DND window."""
    if not dnd_window or not msg_time:
        return False
    try:
        start_str, end_str = dnd_window.split("-")
        msg_hour = datetime.strptime(msg_time.strip(), "%Y-%m-%d %H:%M").hour
        start_h = int(start_str.split(":")[0])
        end_h = int(end_str.split(":")[0])

        if start_h > end_h:  # Overnight DND (e.g., 22:00-07:00)
            return msg_hour >= start_h or msg_hour < end_h
        else:
            return start_h <= msg_hour < end_h
    except (ValueError, IndexError):
        return False


def build_message_context(msg: dict, store: DataStore, user_contexts: dict) -> MessageContext:
    """
    Assemble the full context packet for a single incoming message.

    This combines message fields + user profile + group/business context.
    Media content (unified_text) is set separately by the multimodal normalizer.
    """
    user_id = msg["user_id"]
    conv_type = msg.get("conversation_type", "")

    ctx = MessageContext(
        message_id=msg["message_id"],
        user_id=user_id,
        conversation_type=conv_type,
        message_text=msg.get("message_text", ""),
        media_type=msg.get("media_type", ""),
        media_id=msg.get("media_id", ""),
        forwarded_count=int(msg.get("forwarded_count", 0)),
        created_at=msg.get("created_at", ""),
        sender_user_id=msg.get("sender_user_id", ""),
        group_id=msg.get("group_id", ""),
        business_id=msg.get("business_id", ""),
    )

    # User context (pre-computed)
    ctx.user_ctx = user_contexts.get(user_id, build_user_context(user_id, store))

    # DND check
    ctx.is_in_dnd = is_in_dnd(ctx.created_at, ctx.user_ctx.dnd_window)

    # Conversation-specific context
    if conv_type == "group" and ctx.group_id:
        ctx.group_ctx = build_group_context(
            ctx.group_id, user_id, ctx.sender_user_id, store
        )
    elif conv_type == "business" and ctx.business_id:
        ctx.business_ctx = build_business_context(
            ctx.business_id, user_id, store
        )
    elif conv_type == "personal":
        ctx.sender_exists = ctx.sender_user_id in store.users

    return ctx
