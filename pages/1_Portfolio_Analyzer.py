# pages/1_Portfolio_Analyzer.py
# ============================================
# Portfolio Analyzer – Faz Consulting (v2.1)
# Visual em % (eixos + hovers) • cálculos em decimais
# Multi-ativos • Beta • Sharpe • MDD • Correlação
# Fronteira eficiente • CAPM/SML • Rolling do portfólio
# Contribuição ao risco (RC/MRC) • Export CSV
# ============================================

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# Configuração da página
# -------------------------
st.set_page_config(
    page_title="Portfolio Analyzer – Faz Consulting",
    page_icon="📊",
    layout="wide"
)

# Estilo institucional
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

# -------------------------
# Funções auxiliares
# -------------------------
@st.cache_data(show_spinner=False)
def download_prices(tickers: list[str], benchmark: str, period_years: int) -> pd.DataFrame:
    """Baixa preços ajustados (Adj Close ou Close) para todos os tickers + benchmark."""
    all_tickers = list(dict.fromkeys(tickers + [benchmark]))

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * period_years)

    df = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError("Nenhum dado retornado. Verifique tickers e datas.")

    # Extrai Adj Close ou Close, lidando com MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            adj = df["Adj Close"]
        elif "Adj Close" in df.columns.get_level_values(1):
            adj = df.xs("Adj Close", level=1, axis=1)
        elif "Close" in df.columns.get_level_values(0):
            adj = df["Close"]
        elif "Close" in df.columns.get_level_values(1):
            adj = df.xs("Close", level=1, axis=1)
        else:
            raise ValueError("Não foi possível encontrar 'Adj Close' ou 'Close' nos dados.")
    else:
        if "Adj Close" in df.columns:
            adj = df["Adj Close"]
        elif "Close" in df.columns:
            adj = df["Close"]
        else:
            raise ValueError("DataFrame não possui colunas 'Adj Close' ou 'Close'.")

    adj = adj[all_tickers].dropna(how="all")
    return adj


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos simples diários."""
    return prices.pct_change().dropna(how="all")


def beta_numpy(r_asset: pd.Series, r_bench: pd.Series) -> float:
    """Beta via numpy.cov, robusto."""
    r_a, r_b = r_asset.align(r_bench, join="inner")
    x = r_a.values.astype(float)
    y = r_b.values.astype(float)
    if len(x) < 2 or np.var(y) == 0:
        return float("nan")
    cov = np.cov(x, y)
    return float(cov[0, 1] / cov[1, 1])


def dd_series(prices: pd.Series) -> pd.Series:
    peak = prices.cummax()
    return prices / peak - 1.0


def fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x * 100:.{digits}f}%"


def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}"


def sample_portfolios(mu: np.ndarray, cov: np.ndarray, rf: float, n_ports: int = 5000):
    """
    Gera amostra de portfolios aleatórios (long-only) para aproximar a fronteira eficiente.
    Retorna DataFrame com colunas: Retorno, Vol, Sharpe e matriz de pesos (n_ports x n_assets).
    """
    n_assets = len(mu)
    weights = np.random.dirichlet(np.ones(n_assets), size=n_ports)  # (n_ports, n_assets)

    rets = weights @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    sharpe = np.where(vols > 0, (rets - rf) / vols, np.nan)

    df = pd.DataFrame({
        "Retorno": rets,
        "Vol": vols,
        "Sharpe": sharpe
    })
    return df, weights


def build_efficient_frontier(df_ports: pd.DataFrame) -> pd.DataFrame:
    """Aproxima a fronteira eficiente: ordena por Vol e mantém pontos com maior retorno acumulado."""
    df_sorted = df_ports.sort_values("Vol").reset_index(drop=True)
    best_ret = -np.inf
    frontier_rows = []
    for _, row in df_sorted.iterrows():
        if row["Retorno"] > best_ret:
            best_ret = row["Retorno"]
            frontier_rows.append(row)
    df_frontier = pd.DataFrame(frontier_rows)
    return df_frontier


def rolling_beta_series(r_port: pd.Series, r_bench: pd.Series, window: int) -> pd.Series:
    r_p, r_b = r_port.align(r_bench, join="inner")
    cov = r_p.rolling(window).cov(r_b)
    var = r_b.rolling(window).var()
    beta = cov / var
    return beta.dropna()


def rolling_vol_series(r: pd.Series, window: int) -> pd.Series:
    return r.rolling(window).std(ddof=0) * np.sqrt(252.0)


def rolling_sharpe_series(r: pd.Series, rf_annual: float, window: int) -> pd.Series:
    mu = r.rolling(window).mean() * 252.0
    sig = r.rolling(window).std(ddof=0) * np.sqrt(252.0)
    sh = (mu - rf_annual) / sig
    return sh.replace([np.inf, -np.inf], np.nan).dropna()


def risk_contributions(cov_ann: np.ndarray, weights: np.ndarray):
    """
    RC / MRC para volatilidade:
    - sigma_p = sqrt(w' Σ w)
    - MRC = (Σ w) / sigma_p
    - RC = w * MRC
    """
    w = weights.reshape(-1, 1)
    sigma_p = float(np.sqrt((w.T @ cov_ann @ w)[0, 0]))
    if sigma_p == 0:
        mrc = np.zeros_like(w).flatten()
        rc = np.zeros_like(w).flatten()
    else:
        mrc = (cov_ann @ w / sigma_p).flatten()
        rc = (w.flatten() * mrc)
    rc_pct = rc / sigma_p if sigma_p > 0 else np.zeros_like(rc)
    return sigma_p, mrc, rc, rc_pct


# -------------------------
# UI – Cabeçalho
# -------------------------
st.title("📊 Portfolio Analyzer — Faz Consulting")
st.caption("Análise multi-ativos • Risco/Retorno • Beta • Correlação • Fronteira Eficiente • CAPM • RC/MRC")

# -------------------------
# Sidebar – parâmetros
# -------------------------
with st.sidebar:
    st.header("Configuração do portfólio")

    tickers_str = st.text_input(
        "Tickers (separados por vírgula)",
        value="BND, AGG, SPY, VEA, IAU",
        help="Use os tickers do Yahoo Finance (ex.: BND, SPY, BOVA11.SA, PETR4.SA)"
    )
    weights_str = st.text_input(
        "Pesos (mesma ordem, somando ~1.0)",
        value="0.35, 0.25, 0.20, 0.15, 0.05",
        help="Ex.: 0.4, 0.3, 0.3"
    )
    benchmark = st.text_input(
        "Benchmark",
        value="SPY",
        help="Ticker de referência para beta e comparação."
    )

    period_years = st.number_input(
        "Janela histórica (anos)",
        min_value=1,
        max_value=15,
        value=2,
        step=1
    )

    rf_annual = st.number_input(
        "Taxa livre de risco anual (%)",
        min_value=-5.0,
        max_value=20.0,
        value=0.0,
        step=0.25
    ) / 100.0

    st.markdown("---")
    n_ports_sample = st.number_input(
        "Nº de portfolios aleatórios (fronteira)",
        min_value=500,
        max_value=20000,
        value=5000,
        step=500,
        help="Usado para aproximar a fronteira eficiente (amostragem long-only)."
    )

    roll_window = st.number_input(
        "Janela rolling (dias) p/ métricas do portfólio",
        min_value=60,
        max_value=504,
        value=252,
        step=21
    )

    st.markdown("---")
    run_btn = st.button("▶️ Rodar análise", type="primary")

# -------------------------
# Execução principal
# -------------------------
if run_btn:
    try:
        # 1) Parse de tickers e pesos
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        weights_list = [float(x.strip()) for x in weights_str.replace(";", ",").split(",") if x.strip()]

        if len(tickers) == 0:
            st.error("Informe ao menos um ticker.")
            st.stop()
        if len(tickers) != len(weights_list):
            st.error("Número de pesos diferente do número de tickers.")
            st.stop()

        weights = np.array(weights_list, dtype=float)
        w_sum = weights.sum()
        if w_sum <= 0:
            st.error("Soma dos pesos deve ser > 0.")
            st.stop()
        weights = weights / w_sum

        # 2) Download de preços e retornos
        prices = download_prices(tickers, benchmark, period_years)
        returns = compute_returns(prices)

        r_bench = returns[benchmark].dropna()

        # 3) Estatísticas individuais
        stats_rows = []
        for i, t in enumerate(tickers):
            r = returns[t].dropna()
            r_a, r_b = r.align(r_bench, join="inner")
            if r_a.empty:
                continue

            ret_ann = float(r_a.mean() * 252.0)
            vol_ann = float(r_a.std(ddof=0) * np.sqrt(252.0))
            sharpe = (ret_ann - rf_annual) / vol_ann if vol_ann > 0 else np.nan
            mdd = float(dd_series(prices[t].loc[r_a.index]).min())

            if t == benchmark:
                beta_val = 1.0
            else:
                beta_val = beta_numpy(r_a, r_b)

            stats_rows.append(
                {
                    "Ticker": t,
                    "Peso": weights[i],
                    "Retorno Anual": ret_ann,
                    "Volatilidade": vol_ann,
                    "Sharpe": sharpe,
                    "Beta vs Bench": beta_val,
                    "Max Drawdown": mdd,
                }
            )

        if not stats_rows:
            st.error("Não foi possível calcular estatísticas. Verifique os tickers.")
            st.stop()

        df_stats = pd.DataFrame(stats_rows)
        df_stats["Peso_%"] = df_stats["Peso"] * 100.0

        # 4) Estatísticas do portfólio
        r_port = (returns[tickers] * weights).sum(axis=1).dropna()
        r_port, r_bench_aligned = r_port.align(r_bench, join="inner")

        port_ret_ann = float(r_port.mean() * 252.0)
        port_vol_ann = float(r_port.std(ddof=0) * np.sqrt(252.0))
        port_sharpe = (port_ret_ann - rf_annual) / port_vol_ann if port_vol_ann > 0 else np.nan
        port_beta = beta_numpy(r_port, r_bench_aligned)

        # MDD do portfólio (em $1)
        px_port = (1.0 + r_port).cumprod()
        port_dd = dd_series(px_port)
        port_mdd = float(port_dd.min())

        # Crescimento de $1
        port_cum = px_port / px_port.iloc[0]
        bench_cum = (1.0 + r_bench_aligned).cumprod()
        bench_cum = bench_cum / bench_cum.iloc[0]

        # Matriz de correlação entre os ativos do portfólio
        corr = returns[tickers].corr()

        # -------------------------
        # KPIs principais
        # -------------------------
        st.markdown("## 📌 Métricas principais do portfólio")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retorno anual", fmt_pct(port_ret_ann))
        c2.metric("Volatilidade anual", fmt_pct(port_vol_ann))
        c3.metric("Sharpe", fmt_num(port_sharpe))
        c4.metric(f"Beta vs {benchmark}", fmt_num(port_beta))

        c5, c6 = st.columns(2)
        c5.metric("Max Drawdown", fmt_pct(port_mdd))
        c6.metric("Nº de ativos", str(len(tickers)))

        # -------------------------
        # Tabela – Estatísticas individuais
        # -------------------------
        st.markdown("### Estatísticas individuais dos ativos")

        df_display = df_stats.copy()
        df_display["Peso"] = df_display["Peso_%"].map(lambda x: f"{x:.1f}%")
        df_display["Retorno Anual"] = df_display["Retorno Anual"].map(lambda x: f"{x*100:.2f}%")
        df_display["Volatilidade"] = df_display["Volatilidade"].map(lambda x: f"{x*100:.2f}%")
        df_display["Sharpe"] = df_display["Sharpe"].map(lambda x: f"{x:.2f}")
        df_display["Beta vs Bench"] = df_display["Beta vs Bench"].map(
            lambda x: f"{x:.2f}" if np.isfinite(x) else "N/A"
        )
        df_display["Max Drawdown"] = df_display["Max Drawdown"].map(lambda x: f"{x*100:.2f}%")
        df_display = df_display[
            ["Ticker", "Peso", "Retorno Anual", "Volatilidade", "Sharpe", "Beta vs Bench", "Max Drawdown"]
        ]

        st.dataframe(df_display, use_container_width=True)

        # -------------------------
        # Gráfico – Performance acumulada
        # -------------------------
        st.markdown("### Performance acumulada — Portfólio vs Benchmark")

        # customdata com retorno acumulado (= crescimento - 1)
        port_custom = np.stack([(port_cum.values - 1.0)], axis=-1)
        bench_custom = np.stack([(bench_cum.values - 1.0)], axis=-1)

        fig_perf = go.Figure()
        fig_perf.add_trace(
            go.Scatter(
                x=port_cum.index,
                y=port_cum.values,
                mode="lines",
                name="Portfólio",
                customdata=port_custom,
                hovertemplate=(
                    "Data: %{x|%Y-%m-%d}<br>"
                    "Crescimento: %{y:.3f}x<br>"
                    "Retorno: %{customdata[0]:+.2%}<extra></extra>"
                ),
            )
        )
        fig_perf.add_trace(
            go.Scatter(
                x=bench_cum.index,
                y=bench_cum.values,
                mode="lines",
                name=f"Benchmark ({benchmark})",
                customdata=bench_custom,
                hovertemplate=(
                    "Data: %{x|%Y-%m-%d}<br>"
                    "Crescimento: %{y:.3f}x<br>"
                    "Retorno: %{customdata[0]:+.2%}<extra></extra>"
                ),
            )
        )
        fig_perf.update_layout(
            title="Crescimento de $1",
            xaxis_title="Data",
            yaxis_title="Crescimento de $1",
            xaxis=dict(rangeslider=dict(visible=True)),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        # -------------------------
        # Heatmap de correlação
        # -------------------------
        st.markdown("### Matriz de correlação entre ativos")

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            zmin=-1,
            zmax=1
        )
        fig_corr.update_layout(
            xaxis_title="Ticker",
            yaxis_title="Ticker",
            coloraxis_colorbar=dict(title="Correlação")
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # =====================================================
        # 🔹 Fronteira eficiente, GMV, Tangency portfolio
        # =====================================================
        st.markdown("## Fronteira eficiente (Markowitz – long only)")

        mu = df_stats.set_index("Ticker")["Retorno Anual"].reindex(tickers).values
        cov_daily = returns[tickers].cov().values
        cov_ann = cov_daily * 252.0

        df_ports, weights_sample = sample_portfolios(mu, cov_ann, rf_annual, n_ports=int(n_ports_sample))
        df_frontier = build_efficient_frontier(df_ports)

        idx_gmv = int(df_ports["Vol"].idxmin())
        w_gmv = weights_sample[idx_gmv]
        ret_gmv = float(df_ports.loc[idx_gmv, "Retorno"])
        vol_gmv = float(df_ports.loc[idx_gmv, "Vol"])

        idx_tan = int(df_ports["Sharpe"].idxmax())
        w_tan = weights_sample[idx_tan]
        ret_tan = float(df_ports.loc[idx_tan, "Retorno"])
        vol_tan = float(df_ports.loc[idx_tan, "Vol"])
        sharpe_tan = float(df_ports.loc[idx_tan, "Sharpe"])

        st.markdown(
            f"""
            - Portfólio **mínima vol (GMV)**: Ret {fmt_pct(ret_gmv)}, Vol {fmt_pct(vol_gmv)}  
            - Portfólio **máx. Sharpe (Tangency)**: Ret {fmt_pct(ret_tan)}, Vol {fmt_pct(vol_tan)}, Sharpe {fmt_num(sharpe_tan,2)}
            """
        )

        fig_front = go.Figure()

        fig_front.add_trace(
            go.Scatter(
                x=df_ports["Vol"],
                y=df_ports["Retorno"],
                mode="markers",
                marker=dict(size=4, opacity=0.3),
                name="Portf. aleatórios",
                hovertemplate="Vol: %{x:.1%}<br>Ret: %{y:.1%}<extra></extra>",
            )
        )
        fig_front.add_trace(
            go.Scatter(
                x=df_frontier["Vol"],
                y=df_frontier["Retorno"],
                mode="lines",
                line=dict(width=3),
                name="Fronteira eficiente (aprox.)",
                hovertemplate="Vol: %{x:.1%}<br>Ret: %{y:.1%}<extra></extra>",
            )
        )
        fig_front.add_trace(
            go.Scatter(
                x=[port_vol_ann],
                y=[port_ret_ann],
                mode="markers",
                marker=dict(size=12, symbol="diamond"),
                name="Portfólio atual",
                hovertemplate="Vol: %{x:.1%}<br>Ret: %{y:.1%}<extra></extra>",
            )
        )
        fig_front.add_trace(
            go.Scatter(
                x=[vol_gmv],
                y=[ret_gmv],
                mode="markers",
                marker=dict(size=10, symbol="triangle-down"),
                name="GMV (mín vol)",
                hovertemplate="Vol: %{x:.1%}<br>Ret: %{y:.1%}<extra></extra>",
            )
        )
        fig_front.add_trace(
            go.Scatter(
                x=[vol_tan],
                y=[ret_tan],
                mode="markers",
                marker=dict(size=10, symbol="triangle-up"),
                name="Tangency (máx Sharpe)",
                hovertemplate="Vol: %{x:.1%}<br>Ret: %{y:.1%}<extra></extra>",
            )
        )

        fig_front.update_layout(
            title="Risco x Retorno — Fronteira eficiente (amostragem)",
            xaxis_title="Volatilidade anual",
            yaxis_title="Retorno anual",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_front, use_container_width=True)

        # =====================================================
        # 🔹 CAPM / Security Market Line (SML)
        # =====================================================
        st.markdown("## CAPM — Security Market Line (SML)")

        r_bench_ann = float(r_bench_aligned.mean() * 252.0)

        df_capm = df_stats[["Ticker", "Retorno Anual", "Beta vs Bench"]].copy()
        df_capm.loc[len(df_capm)] = {
            "Ticker": "PORTFOLIO",
            "Retorno Anual": port_ret_ann,
            "Beta vs Bench": port_beta
        }

        betas = df_capm["Beta vs Bench"].values.astype(float)
        rets = df_capm["Retorno Anual"].values.astype(float)

        beta_min = np.nanmin(betas) - 0.2
        beta_max = np.nanmax(betas) + 0.2
        grid_beta = np.linspace(beta_min, beta_max, 50)
        sml_ret = rf_annual + grid_beta * (r_bench_ann - rf_annual)

        fig_sml = go.Figure()
        fig_sml.add_trace(
            go.Scatter(
                x=grid_beta,
                y=sml_ret,
                mode="lines",
                name="SML (CAPM)",
                hovertemplate="Beta: %{x:.2f}<br>E[R]: %{y:.1%}<extra></extra>",
            )
        )
        fig_sml.add_trace(
            go.Scatter(
                x=betas,
                y=rets,
                mode="markers+text",
                text=df_capm["Ticker"],
                textposition="top center",
                name="Ativos / Portfólio",
                hovertemplate="Ticker: %{text}<br>Beta: %{x:.2f}<br>Ret: %{y:.1%}<extra></extra>",
            )
        )
        fig_sml.update_layout(
            title=f"Security Market Line — rf={fmt_pct(rf_annual)}, E[Rm]={fmt_pct(r_bench_ann)}",
            xaxis_title="Beta vs Benchmark",
            yaxis_title="Retorno anual realizado",
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_sml, use_container_width=True)

        # =====================================================
        # 🔹 Rolling métricas do portfólio
        # =====================================================
        st.markdown("## Rolling métricas do portfólio")

        rb = rolling_beta_series(r_port, r_bench_aligned, window=int(roll_window))
        rv = rolling_vol_series(r_port, window=int(roll_window))
        rs = rolling_sharpe_series(r_port, rf_annual=rf_annual, window=int(roll_window))

        gR1, gR2 = st.columns(2)
        with gR1:
            fig_rbeta = go.Figure()
            fig_rbeta.add_trace(
                go.Scatter(
                    x=rb.index,
                    y=rb.values,
                    mode="lines",
                    name="Rolling Beta",
                    hovertemplate="Data: %{x|%Y-%m-%d}<br>Beta: %{y:.2f}<extra></extra>",
                )
            )
            fig_rbeta.update_layout(
                title=f"Rolling Beta vs {benchmark} (janela={int(roll_window)}d)",
                xaxis_title="Data",
                yaxis_title="Beta",
                xaxis=dict(rangeslider=dict(visible=True))
            )
            st.plotly_chart(fig_rbeta, use_container_width=True)

        with gR2:
            fig_rvol = go.Figure()
            fig_rvol.add_trace(
                go.Scatter(
                    x=rv.index,
                    y=rv.values,
                    mode="lines",
                    name="Rolling Vol (ann)",
                    hovertemplate="Data: %{x|%Y-%m-%d}<br>Vol (ann): %{y:.1%}<extra></extra>",
                )
            )
            fig_rvol.update_layout(
                title=f"Rolling Volatilidade Anualizada (janela={int(roll_window)}d)",
                xaxis_title="Data",
                yaxis_title="Vol (ann)",
                xaxis=dict(rangeslider=dict(visible=True)),
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_rvol, use_container_width=True)

        fig_rsh = go.Figure()
        fig_rsh.add_trace(
            go.Scatter(
                x=rs.index,
                y=rs.values,
                mode="lines",
                name="Rolling Sharpe",
                hovertemplate="Data: %{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>",
            )
        )
        fig_rsh.update_layout(
            title=f"Rolling Sharpe do portfólio (rf={rf_annual*100:.2f}%, janela={int(roll_window)}d)",
            xaxis_title="Data",
            yaxis_title="Sharpe",
            xaxis=dict(rangeslider=dict(visible=True))
        )
        st.plotly_chart(fig_rsh, use_container_width=True)

        # =====================================================
        # 🔹 Contribuição ao risco (RC / MRC)
        # =====================================================
        st.markdown("## Contribuição ao risco (RC / MRC)")

        sigma_p, mrc, rc, rc_pct = risk_contributions(cov_ann, weights)

        df_rc = pd.DataFrame({
            "Ticker": tickers,
            "Peso": weights,
            "Peso_%": weights * 100.0,
            "MRC": mrc,
            "RC": rc,
            "RC_%_Vol": rc_pct * 100.0,  # em %
        })

        df_rc_display = df_rc.copy()
        df_rc_display["Peso"] = df_rc_display["Peso_%"].map(lambda x: f"{x:.1f}%")
        df_rc_display["MRC"] = df_rc_display["MRC"].map(lambda x: f"{x:.4f}")
        df_rc_display["RC"] = df_rc_display["RC"].map(lambda x: f"{x:.4f}")
        df_rc_display["RC_%_Vol"] = df_rc_display["RC_%_Vol"].map(lambda x: f"{x:.1f}%")
        df_rc_display = df_rc_display[["Ticker", "Peso", "MRC", "RC", "RC_%_Vol"]]

        cRC1, cRC2 = st.columns([1.2, 1.8])
        with cRC1:
            st.write("**Tabela de contribuições ao risco**")
            st.dataframe(df_rc_display, use_container_width=True)

        with cRC2:
            fig_rc = go.Figure(
                go.Bar(
                    x=df_rc["RC_%_Vol"],
                    y=df_rc["Ticker"],
                    orientation="h",
                    text=[f"{v:.1f}%" for v in df_rc["RC_%_Vol"]],
                    textposition="outside",
                    hovertemplate="Ativo: %{y}<br>RC: %{x:.1f}%<extra></extra>",
                )
            )
            fig_rc.update_layout(
                title="Contribuição de cada ativo na volatilidade do portfólio",
                xaxis_title="% da volatilidade total",
                yaxis_title="Ticker",
            )
            st.plotly_chart(fig_rc, use_container_width=True)

        # =====================================================
        # 🔹 Exportar resultados (CSV)
        # =====================================================
        st.markdown("## Exportar relatório (CSV)")

        df_export = df_stats.copy()
        df_export["Peso_%"] = df_export["Peso"] * 100.0
        df_export["Retorno_Anual_%"] = df_export["Retorno Anual"] * 100.0
        df_export["Volatilidade_%"] = df_export["Volatilidade"] * 100.0
        df_export["Max_Drawdown_%"] = df_export["Max Drawdown"] * 100.0

        df_export = df_export.merge(
            df_rc[["Ticker", "MRC", "RC", "RC_%_Vol"]],
            on="Ticker",
            how="left"
        )

        port_row = pd.DataFrame(
            [{
                "Ticker": "PORTFOLIO",
                "Peso": 1.0,
                "Peso_%": 100.0,
                "Retorno Anual": port_ret_ann,
                "Retorno_Anual_%": port_ret_ann * 100.0,
                "Volatilidade": port_vol_ann,
                "Volatilidade_%": port_vol_ann * 100.0,
                "Sharpe": port_sharpe,
                "Beta vs Bench": port_beta,
                "Max Drawdown": port_mdd,
                "Max_Drawdown_%": port_mdd * 100.0,
                "MRC": np.nan,
                "RC": np.nan,
                "RC_%_Vol": 100.0,
            }]
        )
        df_export = pd.concat([df_export, port_row], ignore_index=True)

        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Baixar CSV com métricas do portfólio",
            data=csv_bytes,
            file_name="portfolio_analyzer_faz_consulting_v2.csv",
            mime="text/csv"
        )

        st.success("Análise de portfólio concluída com sucesso.")

    except Exception as e:
        st.error(f"Erro ao processar a análise de portfólio: {e}")
