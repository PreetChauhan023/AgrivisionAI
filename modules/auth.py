"""
Module: auth.py
Purpose: Handle user authentication – signup, login, session management.
Credentials are persisted in a local CSV file.
"""

import os
import re
import hashlib
import pandas as pd
import streamlit as st

# Path to the users CSV file
USERS_CSV = "data/users.csv"
COLUMNS   = ["username", "email", "password_hash"]
EMAIL_RE  = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    """Return the SHA-256 hex digest of the password."""
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> pd.DataFrame:
    """Load users from CSV, creating the file if it does not exist."""
    os.makedirs(os.path.dirname(USERS_CSV), exist_ok=True)
    if os.path.exists(USERS_CSV):
        df = pd.read_csv(USERS_CSV)
        # Ensure all expected columns are present
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df
    return pd.DataFrame(columns=COLUMNS)


def _save_users(df: pd.DataFrame) -> None:
    """Persist the users DataFrame to CSV."""
    os.makedirs(os.path.dirname(USERS_CSV), exist_ok=True)
    df.to_csv(USERS_CSV, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def signup_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """
    Register a new user.

    Returns
    -------
    (True, "success message") or (False, "error message")
    """
    username, email = username.strip(), email.strip()
    df = _load_users()

    if username == "" or email == "" or password.strip() == "":
        return False, "All fields are required."

    if not EMAIL_RE.match(email):
        return False, "Please enter a valid email address."

    if username in df["username"].values:
        return False, "Username already exists. Please choose a different one."

    if email in df["email"].values:
        return False, "An account with this email already exists."

    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    new_row = pd.DataFrame([{
        "username"     : username,
        "email"        : email,
        "password_hash": _hash_password(password),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    _save_users(df)
    return True, f"Account created successfully! Welcome, {username} 🌱"


def login_user(username: str, password: str) -> tuple[bool, str]:
    """
    Verify login credentials.

    Returns
    -------
    (True, "success message") or (False, "error message")
    """
    username = username.strip()
    df = _load_users()

    if username == "" or password.strip() == "":
        return False, "Please enter both username and password."

    user_rows = df[df["username"] == username]
    if user_rows.empty:
        return False, "Username not found. Please sign up first."

    stored_hash = user_rows.iloc[0]["password_hash"]
    if stored_hash != _hash_password(password):
        return False, "Incorrect password. Please try again."

    return True, f"Welcome back, {username}! 🌿"


def get_user_info(username: str) -> dict:
    """Return a dict of user info for the given username."""
    df = _load_users()
    rows = df[df["username"] == username]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {"username": row["username"], "email": row["email"]}


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Authentication UI
# ─────────────────────────────────────────────────────────────────────────────

_AUTH_CSS = """
<style>
/* ---- page shell ---- */
[data-testid="stSidebar"], [data-testid="stSidebarNav"],
[data-testid="stHeader"], header, footer { display: none !important; }

.stApp {
    background:
        radial-gradient(900px 500px at 8% 0%,  rgba(74,222,128,.18), transparent 60%),
        radial-gradient(800px 500px at 100% 100%, rgba(250,204,21,.12), transparent 60%),
        #0b1a12;
}
.block-container { padding-top: 3rem !important; max-width: 1100px; }

/* ---- hero (left) ---- */
.av-hero { padding: 1.5rem 1rem 1rem 0; color: #e8f5ec; }
.av-badge {
    display: inline-block; padding: .3rem .8rem; border-radius: 999px;
    background: rgba(74,222,128,.14); border: 1px solid rgba(74,222,128,.35);
    color: #86efac; font-size: .8rem; letter-spacing: .04em; margin-bottom: 1.2rem;
}
.av-hero h1 {
    font-size: 3rem; line-height: 1.08; font-weight: 800; margin: 0 0 1rem 0;
    background: linear-gradient(90deg, #ffffff 30%, #86efac);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.av-hero p.sub { color: #ffffff; font-size: 1.05rem; max-width: 30rem; margin-bottom: 2rem; }
.av-feature {
    display: flex; gap: .9rem; align-items: flex-start; margin-bottom: 1.1rem;
}
.av-feature .ic {
    flex: 0 0 2.4rem; height: 2.4rem; border-radius: .7rem; display: grid; place-items: center;
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1); font-size: 1.2rem;
}
.av-feature { align-items: center; }
.av-feature b { color: #ffffff; display: block; font-size: 1rem; }

/* ---- auth card (right) ---- */
[data-testid="stVerticalBlockBorderWrapper"]:has(.av-card-marker) {
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12); border-radius: 1.25rem;
    padding: 1.2rem 1.1rem .8rem 1.1rem;
    box-shadow: 0 30px 80px rgba(0,0,0,.45);
}
.av-card-title { color: #ffffff; font-size: 1.6rem; font-weight: 800; margin: 0; }
.av-card-sub   { color: #ffffff; font-size: .92rem; margin: .15rem 0 1rem 0; }

/* segmented toggle (radio) */
div[data-testid="stRadio"] > div { gap: 0 !important; background: #eef3ef; padding: .25rem; border-radius: .8rem; }
div[data-testid="stRadio"] label {
    flex: 1; justify-content: center; padding: .45rem .5rem; border-radius: .6rem; cursor: pointer;
    margin: 0 !important;
}
div[data-testid="stRadio"] label > div:first-child { display: none; }   /* hide the radio dot */
div[data-testid="stRadio"] label p { color: #4b5f53; font-weight: 600; font-size: .92rem; }
div[data-testid="stRadio"] label:has(input:checked) { background: #16a34a; }
div[data-testid="stRadio"] label:has(input:checked) p { color: #fff; }

/* inputs */
[data-testid="stForm"] label p, [data-testid="stTextInput"] label p { color: #ffffff !important; font-weight: 600; font-size: .86rem; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #ffffff !important; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; }
[data-testid="stForm"] input {
    background: #f5f8f6 !important; color: #0f2a1a !important; border-radius: .6rem !important;
}
[data-testid="stTextInput"] > div > div { border-radius: .6rem; border-color: #d5e0d9; }
[data-testid="stTextInput"] > div > div:focus-within { border-color: #16a34a; box-shadow: 0 0 0 3px rgba(22,163,74,.2); }

/* primary button */
[data-testid="stFormSubmitButton"] button {
    width: 100%; border: none; border-radius: .7rem; padding: .65rem 1rem; font-weight: 700;
    background: linear-gradient(90deg, #16a34a, #22c55e); color: #fff;
    transition: transform .08s ease, box-shadow .15s ease;
}
[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(22,163,74,.35); color:#fff; }

.av-foot { text-align: center; color: #ffffff; font-size: .78rem; margin-top: .4rem; }
</style>
"""

_HERO_HTML = """
<div class="av-hero">
  <div class="av-badge">🌾 AI for modern farming</div>
  <h1>AgriVision&nbsp;AI</h1>
  <p class="sub">Smarter decisions for every season. Sign in to analyse your crops,
  track insights and get recommendations backed by data.</p>

  <div class="av-feature"><div class="ic">🔍</div>
    <div><b>Instant crop analysis</b></div></div>
  <div class="av-feature"><div class="ic">📈</div>
    <div><b>Track your progress</b></div></div>
  <div class="av-feature"><div class="ic">🌱</div>
    <div><b>Actionable recommendations</b></div></div>
</div>
"""


def _password_hint(pw: str) -> str:
    """Tiny strength label shown after a failed signup attempt."""
    score = sum([len(pw) >= 8, any(c.isdigit() for c in pw),
                 any(c.isupper() for c in pw), any(not c.isalnum() for c in pw)])
    return ["Weak", "Weak", "Fair", "Good", "Strong"][score]


def show_auth_page() -> None:
    """
    Render the combined Login / Signup page.
    Sets st.session_state.logged_in and st.session_state.username on success.
    """
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)

    # Allow a successful signup to flip the toggle back to Login (must happen
    # before the radio widget is created).
    if "_auth_next_mode" in st.session_state:
        st.session_state["auth_mode"] = st.session_state.pop("_auth_next_mode")
    st.session_state.setdefault("auth_mode", "Login")

    left, right = st.columns([1.15, 1], gap="large")

    # ── LEFT: branding ───────────────────────────────────────────────────────
    with left:
        st.markdown(_HERO_HTML, unsafe_allow_html=True)

    # ── RIGHT: auth card ─────────────────────────────────────────────────────
    with right:
        with st.container(border=True):
            st.markdown('<span class="av-card-marker"></span>', unsafe_allow_html=True)

            mode = st.radio(
                "Mode", ["Login", "Sign Up"], key="auth_mode",
                horizontal=True, label_visibility="collapsed",
            )

            if "_auth_flash" in st.session_state:
                st.success(st.session_state.pop("_auth_flash"))

            # ── LOGIN ────────────────────────────────────────────────────────
            if mode == "Login":
                st.markdown('<p class="av-card-title">Welcome back</p>'
                            '<p class="av-card-sub">Log in to continue to your dashboard.</p>',
                            unsafe_allow_html=True)

                with st.form("login_form", border=False):
                    username = st.text_input("Username", placeholder="e.g. farmer_joe", key="login_username")
                    password = st.text_input("Password", type="password",
                                             placeholder="Your password", key="login_password")
                    submitted = st.form_submit_button("Log in")

                if submitted:
                    success, msg = login_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username  = username.strip()
                        st.rerun()
                    else:
                        st.error(msg)

            # ── SIGN UP ──────────────────────────────────────────────────────
            else:
                st.markdown('<p class="av-card-title">Create your account</p>'
                            '<p class="av-card-sub">It takes less than a minute.</p>',
                            unsafe_allow_html=True)

                with st.form("signup_form", border=False):
                    new_username = st.text_input("Username", placeholder="Choose a username", key="signup_username")
                    new_email    = st.text_input("Email", placeholder="you@example.com", key="signup_email")
                    c1, c2 = st.columns(2)
                    with c1:
                        new_password = st.text_input("Password", type="password",
                                                     placeholder="Min. 6 characters", key="signup_password")
                    with c2:
                        confirm_pass = st.text_input("Confirm password", type="password",
                                                     placeholder="Repeat password", key="signup_confirm")
                    submitted = st.form_submit_button("Create account")

                if submitted:
                    if new_password != confirm_pass:
                        st.error("Passwords do not match.")
                    else:
                        success, msg = signup_user(new_username, new_email, new_password)
                        if success:
                            st.session_state["_auth_flash"]     = msg + " Please log in."
                            st.session_state["_auth_next_mode"] = "Login"
                            st.rerun()
                        else:
                            st.error(msg)
                            if new_password and "Password" in msg:
                                st.caption(f"Password strength: {_password_hint(new_password)}")

                    st.markdown('<p class="av-foot">🔒 Your credentials are stored locally and hashed.</p>',
                        unsafe_allow_html=True)