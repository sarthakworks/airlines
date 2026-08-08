"""India Flight Intelligence — Streamlit app.

Left-panel navigation:
  💬 Ask (chat)      — natural-language assistant (default landing view)
  🔎 Route Explorer  — pick a route → ranked airlines, best time, booking curve, demand
  📊 Sector Issues   — route reliability proxy, complaint categories, busiest routes

Run:  streamlit run dashboard.py --server.port 8502
"""
from __future__ import annotations

import streamlit as st

import chat_agent
import predict as P
import route_analytics as RA
import schedules as SCH

st.set_page_config(page_title="India Flight Intelligence", page_icon="🛫", layout="wide")

CITIES = P.cities()
DEP_TIMES = P.artifacts()[2]["departure_times"]
m = P.metrics()

CHAT, EXPLORER, ISSUES, SCHED = ("💬 Ask (chat)", "🔎 Route Explorer",
                                 "📊 Sector Issues", "🕑 Schedules")

SUGGESTIONS = [
    "Cheapest reliable morning Delhi to Mumbai booked 3 weeks out",
    "Best time of day to fly Bangalore to Delhi?",
    "How far ahead should I book Chennai to Kolkata?",
    "Which airline is most on-time?",
    "What are the most common complaints?",
    "What's the worst route?",
    "Busiest routes in India?",
]


# ------------------------------------------------------------------- views
def render_chat():
    st.subheader("💬 Ask the flight assistant")
    st.caption("Fares, best airline, time of day, booking window, reliability, complaints, "
               "routes — in plain English. (Fares are relative 2022 estimates, not live prices.)")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    typed = st.chat_input("Ask about flights…")
    query = typed or st.session_state.pop("pending_q", None)
    if query:
        st.session_state.chat.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                ans = chat_agent.answer(query)
            st.markdown(ans)
        st.session_state.chat.append({"role": "assistant", "content": ans})


def render_explorer(source, dest, travel_class, dep_time, days_left, rel_w):
    if source == dest:
        st.warning("Pick two different cities in the left panel.")
        return
    st.subheader(f"✈️ {source} → {dest}  ·  {travel_class}  ·  book {days_left} days ahead")

    ranked = P.rank_route(source, dest, days_left=days_left, travel_class=travel_class,
                          departure_time=dep_time, price_weight=1 - rel_w, reliability_weight=rel_w)
    if ranked.empty:
        st.error("No fare data for this route (dataset covers the 6 major metros).")
        return

    times = P.best_time(source, dest, days_left=days_left, travel_class=travel_class)
    curve = P.booking_curve(source, dest, travel_class=travel_class, dep_time=dep_time)
    cc = P.class_compare(source, dest)
    best_days = int(curve.loc[curve["avg_price"].idxmin(), "days_left"])

    top = ranked.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Top pick", top["airline"], f"₹{top['pred_price']:,} · OTP {top['otp']}%")
    k2.metric("Cheapest time to fly", times.iloc[0]["time"], f"₹{times.iloc[0]['avg_price']:,.0f}")
    k3.metric("Cheapest booking window", f"{best_days} days ahead", f"₹{curve['avg_price'].min():,.0f}")
    if cc["Economy"] and cc["Business"]:
        k4.metric("Business vs Economy", f"{cc['Business']/cc['Economy']:.1f}×",
                  f"₹{cc['Business']:,.0f} vs ₹{cc['Economy']:,.0f}")

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### 🏆 Best airlines for this route")
        st.caption("Composite score blends predicted fare (lower = better) with on-time reliability.")
        st.dataframe(ranked[["airline", "pred_price", "otp", "score", "note"]].rename(
            columns={"pred_price": "predicted ₹", "otp": "on-time %", "note": ""}),
            hide_index=True, width="stretch")
        st.bar_chart(ranked.set_index("airline")["score"], height=220)
    with right:
        st.markdown("#### ⏰ Cheapest time of day")
        st.bar_chart(times.set_index("time")["avg_price"], height=180)
        st.markdown("#### 📅 Fare vs booking window")
        st.line_chart(curve.set_index("days_left")["avg_price"], height=180)

    st.divider()
    st.markdown("#### 🎫 Individual flights on this route")
    st.caption("Actual flight numbers with time-of-day, stops, estimated fare (~your booking "
               "window), on-time %, and composite score. The dataset has time-of-day buckets "
               "only — no exact clock times or day-of-week.")
    fl = P.rank_flights(source, dest, days_left=days_left, travel_class=travel_class,
                        top_n=15, price_weight=1 - rel_w, reliability_weight=rel_w)
    if not fl.empty:
        st.dataframe(
            fl[["flight", "airline", "departure_time", "stops", "duration", "price", "otp", "score"]]
            .rename(columns={"departure_time": "time of day", "price": "fare ₹", "otp": "on-time %"}),
            hide_index=True, width="stretch")
    else:
        st.info("No individual-flight data for this route.")

    total, series = P.route_demand(source, dest)
    if total:
        st.divider()
        st.markdown(f"#### 📈 Route demand · **{total/1e6:,.1f}M** passengers (DGCA, all-time)")
        if not series.empty:
            st.line_chart(series["total_pax"], height=200)


def render_sector_issues():
    st.subheader("📊 Route & sector issues")
    st.markdown("#### 🧭 Route reliability (proxy)")
    st.caption("On-time record of the airlines operating each route, weighted by their share. "
               "Metro routes share similar airline mixes, so differences are small — a proxy, "
               "not measured per-route delays.")
    rr = RA.route_reliability()
    if not rr.empty:
        st.dataframe(rr.rename(columns={"reliability_otp": "on-time % (proxy)"}),
                     hide_index=True, width="stretch")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🧳 Complaint categories (2021–26)")
        cc = RA.complaint_categories()
        if not cc.empty:
            st.bar_chart(cc.set_index("category")["total"], height=280)
        st.caption("'Others' is a catch-all; Refunds & Baggage are the top specific issues.")
    with c2:
        st.markdown("#### 📈 Busiest routes (all 1,933 city-pairs)")
        br = RA.busiest_routes(n=10)
        if not br.empty:
            st.bar_chart(br.set_index("route")["million_pax"], height=280)
        st.caption("Million passengers, all-time in the DGCA data.")


def render_schedules(origin, dest, day, cls):
    st.subheader(f"🕑 Scheduled flights · {origin} → {dest}")
    st.caption("Real flight numbers with exact departure/arrival times and (where available) "
               "actual prices, from the added historical datasets (2018–2024, ~84 cities incl. "
               "non-metros). Times and prices are historical/indicative, not live.")
    summ = SCH.route_summary(origin, dest)
    if not summ:
        st.info("No scheduled-flight records for this city pair — try another.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{summ['records']:,}")
    if summ.get("price_median"):
        c2.metric("Median fare", f"₹{summ['price_median']:,}", f"min ₹{summ['price_min']:,}")
    c3.metric("Airlines seen", len(summ["airlines"]))
    df = SCH.list_scheduled_flights(origin, dest, day_of_week=None if day == "Any" else day,
                                    travel_class=None if cls == "Any" else cls, limit=50)
    if df.empty:
        st.info("No flights match those filters.")
        return
    st.dataframe(df.rename(columns={"flight_no": "flight", "dep_time": "dep", "arr_time": "arr",
                                    "price": "fare ₹", "source_ds": "source",
                                    "observations": "obs"}),
                 hide_index=True, width="stretch")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🛫 Flight Intelligence")
    st.caption("Real Indian airline data · DGCA + 300k fares")
    view = st.radio("View", [CHAT, EXPLORER, ISSUES, SCHED], index=0, label_visibility="collapsed")
    st.divider()

    ctrl, sctrl = {}, {}
    if view == EXPLORER:
        ctrl["source"] = st.selectbox("From", CITIES, index=CITIES.index("Delhi"))
        ctrl["dest"] = st.selectbox("To", CITIES, index=CITIES.index("Mumbai"))
        ctrl["travel_class"] = st.radio("Class", ["Economy", "Business"], horizontal=True)
        ctrl["dep_time"] = st.selectbox("Departure time", DEP_TIMES, index=DEP_TIMES.index("Morning"))
        ctrl["days_left"] = st.slider("Book this many days ahead", 1, 45, 15)
        ctrl["rel_w"] = st.slider("Priority:  cheaper  ⟵⟶  reliable", 0.0, 1.0, 0.5, 0.1,
                                  help="0 = rank purely by price · 1 = purely by reliability")
    elif view == CHAT:
        st.markdown("**💡 Try a question**")
        for i, s in enumerate(SUGGESTIONS):
            if st.button(s, key=f"sg{i}"):
                st.session_state["pending_q"] = s
    elif view == SCHED:
        scities = SCH.available_cities()
        di = scities.index("Delhi") if "Delhi" in scities else 0
        mi = scities.index("Mumbai") if "Mumbai" in scities else min(1, len(scities) - 1)
        sctrl["origin"] = st.selectbox("From", scities, index=di)
        sctrl["dest"] = st.selectbox("To", scities, index=mi)
        sctrl["day"] = st.selectbox("Day of week", ["Any", "Monday", "Tuesday", "Wednesday",
                                                    "Thursday", "Friday", "Saturday", "Sunday"])
        sctrl["cls"] = st.selectbox("Class", ["Any", "Economy", "Business"])

    st.divider()
    st.caption(f"Fare model · MAE ₹{m['mae']:,.0f} · MAPE {m['mape']:.1f}% · R² {m['r2']:.3f}")

# --------------------------------------------------------------------- main
st.title("India Flight Intelligence")
if view == CHAT:
    render_chat()
elif view == EXPLORER:
    render_explorer(**ctrl)
elif view == ISSUES:
    render_sector_issues()
else:
    render_schedules(**sctrl)

st.caption("Sources: DGCA (on-time performance, grievances, city-pair traffic) · "
           "public 300k-row Indian fare dataset (2022 snapshot, 6 metros). "
           "Fares are relative estimates, not live prices. Defunct airlines excluded from ranking.")
