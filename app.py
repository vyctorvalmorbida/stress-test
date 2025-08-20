# app.py
# Aplicação web de Stress Test financeiro – Streamlit
# Agora com Upside/Downside Capture Ratio (frequência configurável).

import io
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# =========================
# Configuração geral
# =========================
st.set_page_config(
    page_title="Stress Test Financeiro",
    page_icon="📉",
    layout="wide"
)

# =========================
# Funções auxiliares
# =========================
@st.cache_data(show_spinner=False)
def download_series(ticker: str, start: str, end: str | None):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"Nenhum dado retornado para {ticker}. Verifique ticker e datas.")
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"][ticker].dropna()
    else:
        close = data["Close"].dropna()
    close.name = ticker
    return close

def pct_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()

def realized_vol_annual(returns_daily: pd.Series) -> float:
    return float(returns_daily.std() * np.sqrt(252.0))

def dd_series(prices: pd.Series) -> pd.Series:
    peak = prices.cummax()
    return prices / peak - 1.0

def rolling_period_return(prices: pd.Series, freq: str = "W-FRI") -> pd.Series:
    if freq == "M":
        freq = "ME"
    px_res = prices.resample(freq).last().dropna()
    return px_res.pct_change().dropna()

def var_es_from_distribution(r: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    r_sorted = np.sort(r)
    idx = max(int(alpha * len(r_sorted)) - 1, 0)
    var = r_sorted[idx]
    es = r_sorted[:idx+1].mean() if idx >= 0 else r_sorted[0]
    return float(var), float(es)

def regress_beta(y: np.ndarray, x: np.ndarray) -> float:
    cov = np.cov(y, x)[0, 1]
    var = np.var(x)
    return float(cov / var) if var != 0 else np.nan

def returns_by_freq(prices: pd.Series, label: str) -> pd.Series:
    if label == "Diário":
        return pct_returns(prices)
    elif label == "Semanal":
        return rolling_period_return(prices, "W-FRI")
    else:
        return rolling_period_return(prices, "ME")

def capture_ratio_geometric(r_asset: pd.Series, r_bench: pd.Series, side: str) -> float:
    r_asset, r_bench = r_asset.align(r_bench, join="inner")
    mask = r_bench > 0 if side == "up" else (r_bench < 0)
    if mask.sum() == 0:
        return float("nan")
    asset_g = (1.0 + r_asset[mask]).prod() - 1.0
    bench_g = (1.0 + r_bench[mask]).prod() - 1.0
    if bench_g == 0:
        return float("nan")
    return float((asset_g / bench_g) * 100.0)

def fig_price(px: pd.Series, ticker: str):
    fig = plt.figure(figsize=(8, 4.2))
    plt.plot(px.index, px.values)
    plt.title(f"{ticker} — Preço histórico")
    plt.xlabel("Data"); plt.ylabel("Preço")
    plt.tight_layout()
    return fig

def fig_drawdown(dd: pd.Series, ticker: str, max_dd: float):
    fig = plt.figure(figsize=(8, 3.8))
    plt.plot(dd.index, dd.values)
    plt.title(f"{ticker} — Drawdown (máx: {max_dd:.1%})")
    plt.xlabel("Data"); plt.ylabel("Drawdown")
    plt.tight_layout()
    return fig

def fig_mc_hist(mc_ret: np.ndarray, var_5: float, var_1: float, horizon_days: int, ticker: str):
    fig = plt.figure(figsize=(7.5, 4.2))
    plt.hist(mc_ret, bins=60)
    plt.axvline(var_5, linestyle="--")
    plt.axvline(var_1, linestyle="--")
    plt.title(f"{ticker} — Monte Carlo ({horizon_days}d)  VaR5%={var_5:.1%}  VaR1%={var_1:.1%}")
    plt.xlabel("Retorno no horizonte"); plt.ylabel("Frequência")
    plt.tight_layout()
    return fig

def fig_scenarios(scenario_results: dict, horizon_days: int, ticker: str):
    labels = list(scenario_results.keys())
    vals = [scenario_results[k] for k in labels]
    order = np.argsort(vals)
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    fig = plt.figure(figsize=(8.6, 5.0))
    pos = np.arange(len(labels))
    plt.barh(pos, vals)
    plt.yticks(pos, labels)
    plt.title(f"{ticker} — Retornos estimados por cenário (h={horizon_days}d)")
    plt.xlabel("Retorno")
    plt.tight_layout()
    return fig

# =========================
# UI — Sidebar
# =========================
st.title("📉 Stress Test Financeiro")
st.caption("Monte Carlo + 3σ + Cenários hipotéticos + Capture Ratios (Yahoo Finance)")

with st.sidebar:
    st.header("Parâmetros")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="ARKK", help="Ex.: AAPL, SPY, PETR4.SA, BOVA11.SA, BTC-USD, ^BVSP")
    start = st.date_input("Data inicial", value=date(2015, 1, 1))
    end_opt = st.checkbox("Usar data final", value=False)
    end = st.date_input("Data final", value=date.today()) if end_opt else None

    st.markdown("---")
    horizon_days = st.number_input("Horizonte (dias úteis)", min_value=1, max_value=252, value=21, step=1)
    mc_paths = st.number_input("Monte Carlo — nº de simulações", min_value=1000, max_value=200000, value=20000, step=1000)
    df_student = st.number_input("Student-t (df — caudas gordas)", min_value=3, max_value=30, value=5, step=1)
    bench = st.text_input("Benchmark para beta/capture", value="SPY", help="Ex.: SPY, BOVA11.SA, ^BVSP")
    cap_freq_label = st.selectbox("Frequência p/ Capture Ratio", ["Mensal", "Semanal", "Diário"], index=0,
                                  help="Padrão de mercado: Mensal")
    st.markdown("---")
    run_btn = st.button("▶️ Rodar Stress Test", type="primary")

# =========================
# Execução
# =========================
if run_btn:
    try:
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d") if isinstance(end, date) else None

        # 1) Dados
        px = download_series(ticker, start_str, end_str)
        px_bench = download_series(bench, start_str, end_str)

        r_d = pct_returns(px)
        r_bench_d = pct_returns(px_bench).reindex(r_d.index).dropna()
        r_d = r_d.loc[r_bench_d.index]  # alinhar

        # 2) Métricas principais
        vol_ann = realized_vol_annual(r_d)
        price_now = float(px.iloc[-1])
        dd = dd_series(px)
        max_dd = float(dd.min())
        beta = regress_beta(r_d.values, r_bench_d.values)

        # 3) Piores janelas
        worst_days = r_d.nsmallest(5)
        r_week = rolling_period_return(px, "W-FRI")
        r_month = rolling_period_return(px, "ME")
        worst_weeks = r_week.nsmallest(5)
        worst_months = r_month.nsmallest(5)

        # 4) 3σ (~1 mês útil)
        sigma_daily = vol_ann / np.sqrt(252.0)
        sigma_h = sigma_daily * np.sqrt(int(horizon_days))
        shock_3sigma = -3.0 * sigma_h

        # 5) Monte Carlo
        mc_daily = np.random.standard_t(df=int(df_student), size=(int(mc_paths), int(horizon_days)))
        mc_daily = mc_daily / np.sqrt(df_student / (df_student - 2))
        mc_daily = mc_daily * sigma_daily
        mc_paths_sum = mc_daily.sum(axis=1)
        mc_ret = np.exp(mc_paths_sum) - 1.0
        var_5, es_5 = var_es_from_distribution(mc_ret, alpha=0.05)
        var_1, es_1 = var_es_from_distribution(mc_ret, alpha=0.01)

        # 6) Cenários hipotéticos
        scenarios_market = {
            f"{bench} -15% (beta)": -0.15,
            f"{bench} -25% (beta)": -0.25,
            f"{bench} -35% (beta)": -0.35,
        }
        scenario_results = {name: beta * sp_ret for name, sp_ret in scenarios_market.items()}
        scenario_results["3σ (≈1 mês)"] = shock_3sigma
        scenario_results["Choque -20%"] = -0.20
        scenario_results["Choque -30%"] = -0.30
        scenario_results["Choque -40%"] = -0.40

        # 7) Capture Ratios
        r_asset_cap = returns_by_freq(px, cap_freq_label)
        r_bench_cap = returns_by_freq(px_bench, cap_freq_label)
        r_asset_cap, r_bench_cap = r_asset_cap.align(r_bench_cap, join="inner")
        ucr = capture_ratio_geometric(r_asset_cap, r_bench_cap, side="up")
        dcr = capture_ratio_geometric(r_asset_cap, r_bench_cap, side="down")

        # =========================
        # Layout — resultados
        # =========================
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Preço atual", f"{price_now:,.2f}")
        c2.metric("Vol anual", f"{vol_ann:.2%}")
        c3.metric(f"Beta vs {bench}", f"{beta:.2f}")
        c4.metric("Max Drawdown", f"{max_dd:.2%}")
        c5.metric(f"Upside Capture ({cap_freq_label[0]})", f"{ucr:.1f}%")
        c6.metric(f"Downside Capture ({cap_freq_label[0]})", f"{dcr:.1f}%")

        st.markdown("### Gráficos")
        g1, g2 = st.columns(2)
        with g1:
            st.pyplot(fig_price(px, ticker))
        with g2:
            st.pyplot(fig_drawdown(dd, ticker, max_dd))

        g3, g4 = st.columns(2)
        with g3:
            st.pyplot(fig_mc_hist(mc_ret, var_5, var_1, int(horizon_days), ticker))
        with g4:
            st.pyplot(fig_scenarios(scenario_results, int(horizon_days), ticker))

        st.markdown("### Piores janelas")
        c7, c8, c9 = st.columns(3)
        with c7:
            st.write("**Piores dias**")
            st.dataframe(worst_days.apply(lambda x: f"{x:.2%}"))
        with c8:
            st.write("**Piores semanas**")
            st.dataframe(worst_weeks.apply(lambda x: f"{x:.2%}"))
        with c9:
            st.write("**Piores meses**")
            st.dataframe(worst_months.apply(lambda x: f"{x:.2%}"))

        # Relatório CSV p/ download
        st.markdown("### Exportar resultados")
        rows = [
            ("Ticker", ticker),
            ("Inicio", start_str),
            ("Fim", end_str if end_str else "Hoje"),
            ("Benchmark", bench),
            ("Preco_atual", float(price_now)),
            ("Vol_anual", float(vol_ann)),
            (f"Beta_{bench}", float(beta)),
            ("Max_Drawdown", float(max_dd)),
            ("VaR5_MC", float(var_5)),
            ("ES5_MC", float(es_5)),
            ("VaR1_MC", float(var_1)),
            ("ES1_MC", float(es_1)),
            (f"Upside_Capture_{cap_freq_label[0]}", float(ucr)),
            (f"Downside_Capture_{cap_freq_label[0]}", float(dcr)),
            (f"Shock_3sigma_{int(horizon_days)}d", float(shock_3sigma)),
        ]
        for k, v in scenario_results.items():
            rows.append((f"Cenario_{k}", float(v)))
        df_report = pd.DataFrame(rows, columns=["Metrica", "Valor"])

        csv_buf = io.StringIO()
        df_report.to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇️ Baixar relatório (CSV)",
            data=csv_buf.getvalue(),
            file_name=f"{ticker.lower()}_relatorio.csv",
            mime="text/csv"
        )

        st.success("Stress Test concluído com sucesso.")

    except Exception as e:
        st.error(f"Erro ao processar: {e}")
