# app.py
# Stress Test Financeiro – Streamlit (versão avançada, corrigida)
# Correção: evitar conflito de nomes entre plotly.express (px) e série de preços.
# - Beta consistente (OLS)
# - Monte Carlo em log-retornos (t-Student) com drift opcional
# - VaR/ES por quantil robusto
# - CAGR, Rolling CAGR, Rolling Beta/Vol/Sharpe
# - Gráficos modernos com Plotly (interativos + animação opcional)
# - Métricas adicionais (Sharpe, Sortino, Calmar, Hit Ratio, Upside/Downside Avg)
# - Estilização custom e CSV ampliado

import io
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

# =========================
# Configuração geral + Estilo
# =========================
st.set_page_config(
    page_title="Stress Test Financeiro",
    page_icon="📉",
    layout="wide"
)

st.markdown("""
<style>
.main, .block-container { max-width: 1400px; }
h1, h2, h3 { letter-spacing: 0.2px; }
div[data-testid="metric-container"] {
  background: linear-gradient(180deg, rgba(245,247,250,0.7) 0%, rgba(240,242,246,0.7) 100%);
  border: 1px solid rgba(0,0,0,0.06); border-radius: 16px; padding: 12px 16px;
  box-shadow: 0 1px 12px rgba(0,0,0,0.04);
}
[data-testid="stHorizontalBlock"] > div { gap: 12px; }
hr { border-top: 1px solid rgba(0,0,0,0.08); }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

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

def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()

def realized_vol_annual(returns_daily: pd.Series) -> float:
    return float(returns_daily.std(ddof=0) * np.sqrt(252.0))

def dd_series(prices: pd.Series) -> pd.Series:
    peak = prices.cummax()
    return prices / peak - 1.0

def rolling_period_return(prices: pd.Series, freq: str = "W-FRI") -> pd.Series:
    if freq == "M":
        freq = "ME"
    px_res = prices.resample(freq).last().dropna()
    return px_res.pct_change().dropna()

def var_es_from_distribution(r: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    q = float(np.quantile(r, alpha, method="linear"))
    es = float(r[r <= q].mean()) if np.any(r <= q) else q
    return q, es

def regress_beta(y: np.ndarray, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.var(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])

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

def _fmt_pct_or_na(x: float, decimals: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:.{decimals}%}"

def _fmt_num_or_na(x: float, decimals: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:.{decimals}f}"

def cagr_from_prices(px_series: pd.Series) -> float:
    n = len(px_series.dropna())
    if n < 2:
        return float("nan")
    return float((px_series.iloc[-1] / px_series.iloc[0]) ** (252.0 / n) - 1.0)

def rolling_cagr(prices: pd.Series, window_days: int = 252) -> pd.Series:
    pxs = prices.dropna()
    px_shift = pxs.shift(window_days)
    rcagr = (pxs / px_shift) ** (252.0 / window_days) - 1.0
    return rcagr.dropna()

def downside_deviation_annual(r_daily: pd.Series, mar: float = 0.0) -> float:
    r = r_daily - (mar / 252.0)
    r_down = r[r < 0]
    if r_down.empty:
        return 0.0
    dd = r_down.std(ddof=0) * np.sqrt(252.0)
    return float(dd)

def rolling_beta(r_asset: pd.Series, r_bench: pd.Series, window: int = 252) -> pd.Series:
    r_a, r_b = r_asset.align(r_bench, join="inner")
    cov = r_a.rolling(window).cov(r_b)
    var = r_b.rolling(window).var()
    beta = cov / var
    return beta.dropna()

def rolling_vol(r: pd.Series, window: int = 252) -> pd.Series:
    return r.rolling(window).std(ddof=0) * np.sqrt(252.0)

def rolling_sharpe(r: pd.Series, rf_annual: float = 0.0, window: int = 252) -> pd.Series:
    mu = r.rolling(window).mean() * 252.0
    sig = r.rolling(window).std(ddof=0) * np.sqrt(252.0)
    sh = (mu - rf_annual) / sig
    return sh.replace([np.inf, -np.inf], np.nan).dropna()

# =========================
# UI — Sidebar
# =========================
st.title("📉 Stress Test — Faz Consulting Asset Management")
st.caption("Monte Carlo (t-Student, log) • 3σ • Cenários beta • Capture Ratios • CAGR & métricas avançadas (Plotly)")

with st.sidebar:
    st.header("Parâmetros")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="ARKK", help="Ex.: AAPL, SPY, PETR4.SA, BOVA11.SA, BTC-USD, ^BVSP")
    bench = st.text_input("Benchmark", value="SPY", help="Ex.: SPY, BOVA11.SA, ^BVSP")
    start = st.date_input("Data inicial", value=date(2015, 1, 1))
    end_opt = st.checkbox("Usar data final", value=False)
    end = st.date_input("Data final", value=date.today()) if end_opt else None

    st.markdown("---")
    horizon_days = st.number_input("Horizonte (dias úteis)", min_value=1, max_value=252, value=21, step=1)
    mc_paths = st.number_input("Monte Carlo — nº de simulações", min_value=1000, max_value=200000, value=20000, step=1000)
    df_student = st.number_input("Student-t (df — caudas gordas)", min_value=3, max_value=30, value=5, step=1)
    use_drift = st.checkbox("Incluir drift (μ) no MC (log-retornos)", value=False)

    st.markdown("---")
    cap_freq_label = st.selectbox("Frequência p/ Capture Ratio", ["Mensal", "Semanal", "Diário"], index=0)
    rf_annual = st.number_input("Taxa livre de risco anual (%)", min_value=-5.0, max_value=20.0, value=0.0, step=0.25) / 100.0
    roll_window = st.number_input("Janela (dias) p/ Rolling métricas", min_value=60, max_value=504, value=252, step=21)

    st.markdown("---")
    animate_cum = st.checkbox("Animar evolução acumulada por ano", value=False, help="Cria frames por ano no gráfico de retorno acumulado.")
    show_advanced = st.checkbox("Mostrar seções avançadas (Rolling, Scatter risco-retorno, etc.)", value=True)

    st.markdown("---")
    run_btn = st.button("▶️ Rodar", type="primary")

# =========================
# Execução
# =========================
if run_btn:
    try:
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d") if isinstance(end, date) else None

        # 1) Dados
        px_asset = download_series(ticker, start_str, end_str)      # <<< renomeado
        px_bench = download_series(bench, start_str, end_str)

        # 2) Retornos (aritm.)
        r_d = pct_returns(px_asset)
        r_bench_d = pct_returns(px_bench).reindex(r_d.index).dropna()
        if r_bench_d.empty:
            raise ValueError("Benchmark sem dados suficientes no período selecionado.")
        r_d = r_d.loc[r_bench_d.index]

        # 3) Métricas principais
        price_now = float(px_asset.loc[r_d.index].iloc[-1])
        vol_ann = realized_vol_annual(r_d)
        dd = dd_series(px_asset.loc[r_d.index])
        max_dd = float(dd.min())
        beta = regress_beta(r_d.values, r_bench_d.values)

        # CAGR e acumulados
        px_aligned = px_asset.loc[r_d.index]
        cagr = cagr_from_prices(px_aligned)
        cum_asset = (1.0 + r_d).cumprod() - 1.0
        cum_bench = (1.0 + r_bench_d).cumprod() - 1.0
        calmar = (cagr / abs(max_dd)) if max_dd != 0 else np.nan

        # Sharpe / Sortino
        sharpe = ((r_d.mean() * 252.0) - rf_annual) / (r_d.std(ddof=0) * np.sqrt(252.0))
        d_down_ann = downside_deviation_annual(r_d, mar=rf_annual)
        sortino = ((r_d.mean() * 252.0) - rf_annual) / d_down_ann if d_down_ann > 0 else np.nan

        # Capture Ratios
        r_asset_cap = returns_by_freq(px_asset, cap_freq_label)
        r_bench_cap = returns_by_freq(px_bench, cap_freq_label)
        r_asset_cap, r_bench_cap = r_asset_cap.align(r_bench_cap, join="inner")
        ucr = capture_ratio_geometric(r_asset_cap, r_bench_cap, side="up")
        dcr = capture_ratio_geometric(r_asset_cap, r_bench_cap, side="down")

        # Piores janelas
        worst_days = r_d.nsmallest(5)
        r_week = rolling_period_return(px_asset, "W-FRI")
        r_month = rolling_period_return(px_asset, "ME")
        worst_weeks = r_week.nsmallest(5)
        worst_months = r_month.nsmallest(5)

        # 4) Base para MC e 3σ: log-retornos
        r_log = log_returns(px_aligned)
        sigma_daily = float(r_log.std(ddof=0))
        mu_daily = float(r_log.mean()) if use_drift else 0.0
        sigma_h = sigma_daily * np.sqrt(int(horizon_days))
        shock_3sigma = -3.0 * sigma_h

        # Guard-rails MC
        mc_paths_int = int(mc_paths)
        horizon_int = int(horizon_days)
        if horizon_int > 126 and mc_paths_int > 100_000:
            st.info("Reduzindo simulações para 100.000 devido ao horizonte longo.")
            mc_paths_int = 100_000

        # Monte Carlo (t-Student) em log-retornos
        mc_daily = np.random.standard_t(df=int(df_student), size=(mc_paths_int, horizon_int))
        mc_daily = mc_daily / np.sqrt(df_student / (df_student - 2))
        mc_daily = mu_daily + mc_daily * sigma_daily
        mc_paths_sum = mc_daily.sum(axis=1)
        mc_ret = np.exp(mc_paths_sum) - 1.0
        var_5, es_5 = var_es_from_distribution(mc_ret, alpha=0.05)
        var_1, es_1 = var_es_from_distribution(mc_ret, alpha=0.01)

        # 5) Cenários
        scenarios_market = {
            f"{bench} -15% (beta)": -0.15,
            f"{bench} -25% (beta)": -0.25,
            f"{bench} -35% (beta)": -0.35,
        }
        scenario_results = {name: float(beta) * sp_ret for name, sp_ret in scenarios_market.items()}
        scenario_results[f"3σ (h={horizon_int}d)"] = float(np.exp(shock_3sigma) - 1.0)
        scenario_results["Choque -20%"] = -0.20
        scenario_results["Choque -30%"] = -0.30
        scenario_results["Choque -40%"] = -0.40

        # 6) Métricas mensais auxiliares
        r_m = rolling_period_return(px_aligned, "ME")
        r_m_bench = rolling_period_return(px_bench.loc[px_aligned.index], "ME")
        hit_ratio = (r_m > 0).mean() if not r_m.empty else np.nan
        avg_up = r_m[r_m > 0].mean() if np.any(r_m > 0) else np.nan
        avg_down = r_m[r_m < 0].mean() if np.any(r_m < 0) else np.nan
        corr_daily = float(r_d.corr(r_bench_d))

        # =========================
        # KPIs
        # =========================
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Preço atual", f"{price_now:,.2f}")
        c2.metric("CAGR", _fmt_pct_or_na(cagr, 2))
        c3.metric("Vol anual", _fmt_pct_or_na(vol_ann, 2))
        c4.metric("Max Drawdown", _fmt_pct_or_na(max_dd, 2))
        c5.metric(f"Sharpe (ann, rf={rf_annual*100:.2f}%)", _fmt_num_or_na(sharpe, 2))
        c6.metric("Calmar (CAGR/|MDD|)", _fmt_num_or_na(calmar, 2))

        c7, c8, c9, c10 = st.columns(4)
        c7.metric(f"Beta vs {bench}", _fmt_num_or_na(beta, 2))
        c8.metric("Sortino (ann)", _fmt_num_or_na(sortino, 2))
        c9.metric(f"Upside Capture ({cap_freq_label[0]})", _fmt_num_or_na(ucr, 1))
        c10.metric(f"Downside Capture ({cap_freq_label[0]})", _fmt_num_or_na(dcr, 1))

        # =========================
        # Gráficos principais
        # =========================
        st.markdown("### Gráficos principais")
        g1, g2 = st.columns(2)

        with g1:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=px_aligned.index, y=px_aligned.values, mode="lines", name=ticker))
            fig_price.update_layout(
                title=f"{ticker} — Preço histórico",
                xaxis_title="Data", yaxis_title="Preço",
                xaxis=dict(rangeslider=dict(visible=True), rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1a", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ))
            )
            st.plotly_chart(fig_price, use_container_width=True)

        with g2:
            dd_aligned = dd.loc[px_aligned.index]
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_aligned.index, y=dd_aligned.values, fill="tozeroy", mode="lines", name="Drawdown"))
            fig_dd.update_layout(
                title=f"{ticker} — Drawdown (mín: {max_dd:.1%})",
                xaxis_title="Data", yaxis_title="Drawdown",
                xaxis=dict(rangeslider=dict(visible=True))
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("### Retorno acumulado (vs. benchmark)")
        if animate_cum:
            years = pd.Index(cum_asset.index.year.unique(), name="Ano")
            df_anim = pd.DataFrame({
                "Data": cum_asset.index.append(cum_bench.index),
                "Retorno": pd.concat([cum_asset, cum_bench]).values,
                "Série": [ticker]*len(cum_asset) + [bench]*len(cum_bench)
            }).dropna()
            df_anim["Ano"] = df_anim["Data"].dt.year
            fig_cum = px.line(df_anim, x="Data", y="Retorno", color="Série", animation_frame="Ano",
                              title="Retorno acumulado (animação por ano)")
        else:
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(x=cum_asset.index, y=cum_asset.values, mode="lines", name=ticker))
            fig_cum.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench.values, mode="lines", name=bench))
            fig_cum.update_layout(title="Retorno acumulado", xaxis_title="Data", yaxis_title="Retorno",
                                  xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig_cum, use_container_width=True)

        # =========================
        # Monte Carlo & Cenários
        # =========================
        st.markdown("### Monte Carlo e Cenários")
        g3, g4 = st.columns(2)

        with g3:
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(x=mc_ret, nbinsx=60, name="MC"))
            fig_mc.add_vline(x=var_5, line_dash="dash", annotation_text=f"VaR5% {var_5:.1%}", annotation_position="top left")
            fig_mc.add_vline(x=var_1, line_dash="dash", annotation_text=f"VaR1% {var_1:.1%}", annotation_position="top left")
            fig_mc.update_layout(title=f"{ticker} — MC (h={horizon_int}d) | VaR5={var_5:.1%}, VaR1={var_1:.1%}",
                                 xaxis_title="Retorno no horizonte", yaxis_title="Frequência")
            st.plotly_chart(fig_mc, use_container_width=True)

        with g4:
            scen_labels = list(scenario_results.keys())
            scen_vals = [scenario_results[k] for k in scen_labels]
            order = np.argsort(scen_vals)
            scen_labels = [scen_labels[i] for i in order]
            scen_vals = [scen_vals[i] for i in order]
            fig_scen = go.Figure(go.Bar(x=scen_vals, y=scen_labels, orientation="h"))
            fig_scen.update_layout(title=f"{ticker} — Retornos estimados por cenário (h={horizon_int}d)",
                                   xaxis_title="Retorno", yaxis_title="")
            st.plotly_chart(fig_scen, use_container_width=True)

        # =========================
        # Piores janelas
        # =========================
        st.markdown("### Piores janelas")
        cA, cB, cC = st.columns(3)
        with cA:
            st.write("**Piores dias**")
            st.dataframe(worst_days.apply(lambda x: f"{x:.2%}"))
        with cB:
            st.write("**Piores semanas**")
            st.dataframe(worst_weeks.apply(lambda x: f"{x:.2%}"))
        with cC:
            st.write("**Piores meses**")
            st.dataframe(worst_months.apply(lambda x: f"{x:.2%}"))

        # =========================
        # Seções avançadas
        # =========================
        if show_advanced:
            st.markdown("## Análises avançadas")

            st.markdown("### Rolling métricas")
            rbeta = rolling_beta(r_d, r_bench_d, window=int(roll_window))
            rvol = rolling_vol(r_d, window=int(roll_window))
            rsh = rolling_sharpe(r_d, rf_annual=rf_annual, window=int(roll_window))
            rcagr = rolling_cagr(px_aligned, window_days=int(roll_window))

            gR1, gR2 = st.columns(2)
            with gR1:
                fig_rbeta = go.Figure()
                fig_rbeta.add_trace(go.Scatter(x=rbeta.index, y=rbeta.values, mode="lines", name="Rolling Beta"))
                fig_rbeta.update_layout(title=f"Rolling Beta vs {bench} (janela={int(roll_window)}d)",
                                        xaxis_title="Data", yaxis_title="Beta", xaxis=dict(rangeslider=dict(visible=True)))
                st.plotly_chart(fig_rbeta, use_container_width=True)

            with gR2:
                fig_rvol = go.Figure()
                fig_rvol.add_trace(go.Scatter(x=rvol.index, y=rvol.values, mode="lines", name="Rolling Vol (ann)"))
                fig_rvol.update_layout(title=f"Rolling Volatilidade Anualizada (janela={int(roll_window)}d)",
                                       xaxis_title="Data", yaxis_title="Vol (ann)", xaxis=dict(rangeslider=dict(visible=True)))
                st.plotly_chart(fig_rvol, use_container_width=True)

            gR3, gR4 = st.columns(2)
            with gR3:
                fig_rsh = go.Figure()
                fig_rsh.add_trace(go.Scatter(x=rsh.index, y=rsh.values, mode="lines", name="Rolling Sharpe"))
                fig_rsh.update_layout(title=f"Rolling Sharpe (rf={rf_annual*100:.2f}%, janela={int(roll_window)}d)",
                                      xaxis_title="Data", yaxis_title="Sharpe", xaxis=dict(rangeslider=dict(visible=True)))
                st.plotly_chart(fig_rsh, use_container_width=True)

            with gR4:
                fig_rcagr = go.Figure()
                fig_rcagr.add_trace(go.Scatter(x=rcagr.index, y=rcagr.values, mode="lines", name="Rolling CAGR"))
                fig_rcagr.update_layout(title=f"Rolling CAGR (janela={int(roll_window)}d)",
                                        xaxis_title="Data", yaxis_title="CAGR", xaxis=dict(rangeslider=dict(visible=True)))
                st.plotly_chart(fig_rcagr, use_container_width=True)

            st.markdown("### Scatter risco-retorno (mensal) vs. benchmark")
            if not r_m.empty and not r_m_bench.empty:
                df_scatter = pd.DataFrame({
                    "Retorno_Ann": r_m.groupby(r_m.index.year).mean() * 12.0,
                    "Vol_Ann": r_m.groupby(r_m.index.year).std(ddof=0) * np.sqrt(12.0),
                }).dropna()
                df_scatter["Série"] = ticker

                df_scatter_b = pd.DataFrame({
                    "Retorno_Ann": r_m_bench.groupby(r_m_bench.index.year).mean() * 12.0,
                    "Vol_Ann": r_m_bench.groupby(r_m_bench.index.year).std(ddof=0) * np.sqrt(12.0),
                }).dropna()
                df_scatter_b["Série"] = bench

                df_sc = pd.concat([df_scatter, df_scatter_b], axis=0)
                df_sc["Ano"] = df_sc.index
                fig_sc = px.scatter(df_sc, x="Vol_Ann", y="Retorno_Ann", color="Série", hover_name="Ano",
                                    trendline="ols", title="Risco (Vol Ann) x Retorno (Ann) — pontos por ano")
                st.plotly_chart(fig_sc, use_container_width=True)
            else:
                st.info("Histórico mensal insuficiente para scatter risco-retorno.")

            st.markdown("### Estatísticas complementares (mensal)")
            df_stats = pd.DataFrame({
                "Métrica": ["Hit Ratio", "Avg Up (mês)", "Avg Down (mês)", "Correlação diária com bench"],
                "Valor": [ _fmt_pct_or_na(hit_ratio,2),
                           _fmt_pct_or_na(avg_up,2),
                           _fmt_pct_or_na(avg_down,2),
                           _fmt_num_or_na(corr_daily,3) ]
            })
            st.dataframe(df_stats, use_container_width=True)

        # =========================
        # Exportar CSV
        # =========================
        st.markdown("### Exportar resultados")
        rows = [
            ("Ticker", ticker),
            ("Inicio", start_str),
            ("Fim", end_str if end_str else "Hoje"),
            ("Benchmark", bench),
            ("Horizonte_dias", float(horizon_int)),
            ("MC_paths", float(mc_paths_int)),
            ("t_df", float(df_student)),
            ("MC_drift_usado", int(use_drift)),
            ("Capture_freq", cap_freq_label),
            ("Roll_window_dias", float(roll_window)),
            ("RF_annual", float(rf_annual)),
            ("Preco_atual", float(price_now)),
            ("CAGR", float(cagr)),
            ("Vol_anual", float(vol_ann)),
            (f"Beta_{bench}", float(beta) if np.isfinite(beta) else np.nan),
            ("Max_Drawdown", float(max_dd)),
            ("Sharpe_ann", float(sharpe)),
            ("Sortino_ann", float(sortino) if np.isfinite(sortino) else np.nan),
            ("Calmar", float(calmar) if np.isfinite(calmar) else np.nan),
            ("Hit_Ratio_mensal", float(hit_ratio) if np.isfinite(hit_ratio) else np.nan),
            ("Avg_Up_mensal", float(avg_up) if np.isfinite(avg_up) else np.nan),
            ("Avg_Down_mensal", float(avg_down) if np.isfinite(avg_down) else np.nan),
            ("Corr_diaria_bench", float(corr_daily) if np.isfinite(corr_daily) else np.nan),
            ("VaR5_MC", float(var_5)),
            ("ES5_MC", float(es_5)),
            ("VaR1_MC", float(var_1)),
            ("ES1_MC", float(es_1)),
            (f"Upside_Capture_{cap_freq_label[0]}", float(ucr) if np.isfinite(ucr) else np.nan),
            (f"Downside_Capture_{cap_freq_label[0]}", float(dcr) if np.isfinite(dcr) else np.nan),
            (f"Shock_3sigma_log_h{horizon_int}d", float(shock_3sigma)),
            (f"Shock_3sigma_ret_h{horizon_int}d", float(np.exp(shock_3sigma) - 1.0)),
        ]
        for k, v in scenario_results.items():
            rows.append((f"Cenario_{k}", float(v)))
        df_report = pd.DataFrame(rows, columns=["Metrica", "Valor"])

        csv_buf = io.StringIO()
        df_report.to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇️ Baixar relatório (CSV)",
            data=csv_buf.getvalue(),
            file_name=f"{ticker.lower()}_relatorio_avancado.csv",
            mime="text/csv"
        )

        st.success("Análise concluída com sucesso.")

        if not np.isfinite(ucr) or not np.isfinite(dcr):
            st.warning("Capture Ratios retornaram N/A: pode faltar histórico/overlap suficiente na frequência escolhida.")

    except Exception as e:
        st.error(f"Erro ao processar: {e}")
