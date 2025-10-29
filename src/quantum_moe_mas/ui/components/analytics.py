"""
Analytics and Visualization Components for Quantum MoE MAS

This module provides comprehensive analytics and visualization components
for the Streamlit dashboard, including MoE routing visualization, ROI tracking,
performance metrics, and system health monitoring.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

from quantum_moe_mas.core.logging_simple import get_logger
from quantum_moe_mas.moe.analytics import RoutingAnalytics, EfficiencyReport
from quantum_moe_mas.moe.expert import Expert, ExpertType

logger = get_logger(__name__)


@dataclass
class AnalyticsConfig:
    """Configuration for analytics components."""
    refresh_interval: int = 30
    max_data_points: int = 1000
    confidence_threshold: float = 0.8
    show_real_time: bool = True
    enable_export: bool = True


class MoERoutingVisualizer:
    """Visualizes MoE routing decisions and expert selection."""
    
    def __init__(self, config: AnalyticsConfig = None):
        self.config = config or AnalyticsConfig()
        self.logger = get_logger(f"{__name__}.MoERoutingVisualizer")
    
    def render_routing_overview(self, routing_data: Dict[str, Any]) -> None:
        """Render MoE routing overview with confidence scores."""
        st.subheader("🧠 MoE Routing Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_queries = routing_data.get('total_queries', 0)
            st.metric(
                "Total Queries",
                f"{total_queries:,}",
                delta=routing_data.get('queries_delta', 0)
            )
        
        with col2:
            avg_confidence = routing_data.get('avg_confidence', 0.0)
            st.metric(
                "Avg Confidence",
                f"{avg_confidence:.1%}",
                delta=f"{routing_data.get('confidence_delta', 0.0):.1%}"
            )
        
        with col3:
            routing_efficiency = routing_data.get('routing_efficiency', 0.0)
            st.metric(
                "Routing Efficiency",
                f"{routing_efficiency:.1%}",
                delta=f"{routing_data.get('efficiency_delta', 0.0):.1%}"
            )
        
        with col4:
            active_experts = routing_data.get('active_experts', 0)
            st.metric(
                "Active Experts",
                active_experts,
                delta=routing_data.get('experts_delta', 0)
            )
        
        # Routing decision visualization
        self._render_routing_decisions(routing_data)
        
        # Expert selection heatmap
        self._render_expert_selection_heatmap(routing_data)
    
    def _render_routing_decisions(self, routing_data: Dict[str, Any]) -> None:
        """Render routing decision flow visualization."""
        st.markdown("### 🎯 Routing Decisions")
        
        # Create routing flow chart
        decisions = routing_data.get('recent_decisions', [])
        if not decisions:
            st.info("No recent routing decisions available.")
            return
        
        # Prepare data for visualization
        df = pd.DataFrame(decisions)
        
        # Confidence score distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_confidence = px.histogram(
                df, 
                x='confidence_score',
                nbins=20,
                title="Confidence Score Distribution",
                labels={'confidence_score': 'Confidence Score', 'count': 'Frequency'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_confidence.update_layout(
                xaxis_title="Confidence Score",
                yaxis_title="Number of Queries",
                showlegend=False
            )
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        with col2:
            # Expert selection frequency
            expert_counts = df['selected_expert'].value_counts()
            fig_experts = px.pie(
                values=expert_counts.values,
                names=expert_counts.index,
                title="Expert Selection Distribution"
            )
            fig_experts.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_experts, use_container_width=True)
        
        # Timeline of routing decisions
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            fig_timeline = px.scatter(
                df,
                x='timestamp',
                y='confidence_score',
                color='selected_expert',
                size='response_time',
                title="Routing Decisions Timeline",
                hover_data=['query_domain', 'quantum_state']
            )
            fig_timeline.update_layout(
                xaxis_title="Time",
                yaxis_title="Confidence Score",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    def _render_expert_selection_heatmap(self, routing_data: Dict[str, Any]) -> None:
        """Render expert selection heatmap by domain and time."""
        st.markdown("### 🔥 Expert Selection Heatmap")
        
        heatmap_data = routing_data.get('expert_heatmap', {})
        if not heatmap_data:
            st.info("No heatmap data available.")
            return
        
        # Convert to DataFrame for heatmap
        domains = list(heatmap_data.keys())
        experts = list(set().union(*[list(domain_data.keys()) for domain_data in heatmap_data.values()]))
        
        # Create matrix
        matrix = []
        for domain in domains:
            row = []
            for expert in experts:
                count = heatmap_data.get(domain, {}).get(expert, 0)
                row.append(count)
            matrix.append(row)
        
        # Create heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=matrix,
            x=experts,
            y=domains,
            colorscale='Blues',
            text=matrix,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig_heatmap.update_layout(
            title="Expert Selection by Domain",
            xaxis_title="Experts",
            yaxis_title="Domains",
            height=400
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)


class ROIDashboard:
    """ROI tracking and marketing analytics dashboard."""
    
    def __init__(self, config: AnalyticsConfig = None):
        self.config = config or AnalyticsConfig()
        self.logger = get_logger(f"{__name__}.ROIDashboard")
    
    def render_roi_overview(self, roi_data: Dict[str, Any]) -> None:
        """Render ROI overview with key metrics."""
        st.subheader("💰 ROI Dashboard")
        
        # Key ROI metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_roi = roi_data.get('total_roi', 0.0)
            st.metric(
                "Total ROI",
                f"${total_roi:,.2f}",
                delta=f"${roi_data.get('roi_delta', 0.0):,.2f}"
            )
        
        with col2:
            icm_per_session = roi_data.get('icm_per_session', 0.0)
            st.metric(
                "ICM/Session",
                f"${icm_per_session:.2f}",
                delta=f"${roi_data.get('icm_delta', 0.0):.2f}",
                help="Incremental Contribution Margin per User Session"
            )
        
        with col3:
            efficiency_gain = roi_data.get('efficiency_gain', 0.0)
            st.metric(
                "Efficiency Gain",
                f"{efficiency_gain:.1%}",
                delta=f"{roi_data.get('efficiency_delta', 0.0):.1%}"
            )
        
        with col4:
            cost_savings = roi_data.get('cost_savings', 0.0)
            st.metric(
                "Cost Savings",
                f"${cost_savings:,.2f}",
                delta=f"${roi_data.get('savings_delta', 0.0):,.2f}"
            )
        
        # ROI trend analysis
        self._render_roi_trends(roi_data)
        
        # Campaign performance
        self._render_campaign_performance(roi_data)
    
    def _render_roi_trends(self, roi_data: Dict[str, Any]) -> None:
        """Render ROI trend analysis charts."""
        st.markdown("### 📈 ROI Trends")
        
        trend_data = roi_data.get('trend_data', [])
        if not trend_data:
            st.info("No trend data available.")
            return
        
        df = pd.DataFrame(trend_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Create subplots for multiple metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROI Over Time', 'Cost vs Revenue', 'Efficiency Gains', 'Session Value'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # ROI over time
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['roi'], name='ROI', line=dict(color='green')),
            row=1, col=1
        )
        
        # Cost vs Revenue
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['cost'], name='Cost', line=dict(color='red')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['revenue'], name='Revenue', line=dict(color='blue')),
            row=1, col=2
        )
        
        # Efficiency gains
        fig.add_trace(
            go.Bar(x=df['date'], y=df['efficiency_gain'], name='Efficiency Gain'),
            row=2, col=1
        )
        
        # Session value
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['session_value'], name='Session Value', 
                      fill='tonexty', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True, title_text="ROI Analytics Dashboard")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_campaign_performance(self, roi_data: Dict[str, Any]) -> None:
        """Render marketing campaign performance metrics."""
        st.markdown("### 🎯 Campaign Performance")
        
        campaigns = roi_data.get('campaigns', [])
        if not campaigns:
            st.info("No campaign data available.")
            return
        
        df_campaigns = pd.DataFrame(campaigns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Campaign ROI comparison
            fig_roi = px.bar(
                df_campaigns,
                x='campaign_name',
                y='roi',
                color='status',
                title="Campaign ROI Comparison",
                labels={'roi': 'ROI ($)', 'campaign_name': 'Campaign'}
            )
            fig_roi.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_roi, use_container_width=True)
        
        with col2:
            # Conversion rates
            fig_conversion = px.scatter(
                df_campaigns,
                x='impressions',
                y='conversions',
                size='spend',
                color='roi',
                title="Conversion Analysis",
                labels={'impressions': 'Impressions', 'conversions': 'Conversions'}
            )
            st.plotly_chart(fig_conversion, use_container_width=True)


class PerformanceCharts:
    """Performance monitoring and efficiency tracking charts."""
    
    def __init__(self, config: AnalyticsConfig = None):
        self.config = config or AnalyticsConfig()
        self.logger = get_logger(f"{__name__}.PerformanceCharts")
    
    def render_performance_overview(self, performance_data: Dict[str, Any]) -> None:
        """Render performance overview with key metrics."""
        st.subheader("⚡ Performance Metrics")
        
        # Performance KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_latency = performance_data.get('avg_latency', 0.0)
            st.metric(
                "Avg Latency",
                f"{avg_latency:.2f}s",
                delta=f"{performance_data.get('latency_delta', 0.0):.2f}s",
                delta_color="inverse"
            )
        
        with col2:
            throughput = perforeport create PDF tion wouldta  # Implemen")
      y!ssfull succetedort generas("Repsucces   st.
     ort."""ics repive analytomprehensate c"Gener""       
 ) -> None:t[str, Any]data: Diclf, eport(se_generate_r 
    def es
   ata filrt dxpold eentation wou    # Implem   
 lly!")essfuucc exported sess("Dataucc      st.s"
  ""V/JSON.s CSa aaw datxport r""E
        "one: N-> Any]) tr,ct[s Diata:ata(self, df _export_dde  
    G/PDF
  s as PN save charton wouldementati  # Impl!")
      lyssfulsucceed rts exports("Chast.succes      ""
  s images."ort charts aExp     """ None:
   r, Any]) ->Dict[stelf, data: (sort_chartsdef _exp  a)
    
  report(datf._generate_       sel      e):
   dth=Trucontainer_wit", use_orRep📄 Generate button("f st.      i
        with col3:
           )
   (datataexport_daf._         sel      ):
 rueh=Ttainer_widt", use_conataport D("📋 Ex st.button     if:
        col2th wi       
       
 harts(data)._export_c       self    
     th=True):widiner_e_conta", usCharts Export button("📊   if st.:
         with col1      
        3)
  umns( st.colol3 =ol2, c    col1, c    
        
")ionsort Opt"### 📤 Expmarkdown(     st."
   "cs data."ytifor analoptions er export Rend  """     e:
 Any]) -> Non Dict[str, a:ns(self, datptioer_export_o    def rend 
")
   Exporterlytics_}.Ana"{__name__logger(fger = get self.log
       it__(self): def __in
   "
    .""reportsand ytics data r anality fonalport functio """Exrter:
   xpolyticsEss Anacla

       )

 =Trueiner_widthta    use_con
        e']],orll_sc 'overafaction',satisser_', 'ubilityeliay', 'rficienc'cost_ef                
      ed', , 'spe 'accuracy'_type',me', 'expertnaexpert_[' df_ranked[       
    .dataframe(
        st
        =False)ngscendire', aall_scoues('over.sort_valperfnked = df__ra
        df   
            ).1
 tion'] * 0atisfacrf['user_spe      df_    +
  ty'] * 0.2 abilielidf_perf['r         .2 +
   * 0] iciency'st_eff_perf['co       df.2 +
     speed'] * 0erf['  df_p       
    +y'] * 0.3f['accurac     df_per        = (
core']overall_sperf['        df_ore
overall sculate lc   # Ca
          s")
   ance Ranking Performwn("#### 📊kdo      st.martable
  king ce ranorman      # Perf     
  =True)
   r_widthntainedar, use_coart(fig_raotly_ch  st.pl     
               )
"
  rtRadar Chaance rt Performpe title="Ex          end=True,
      showleg
           )),           1]
 range=[0,                   le=True,
  isib    v                is=dict(
dialax      ra  
        olar=dict( p     
      layout(r.update_   fig_rada       
  ))
              
  e=expert nam               ,
toself'='ll  fi       ,
       0]]s[etrics + [mictheta=metr          ,
      alues    r=v        
    atterpolar(o.Scrace(gr.add_trada   fig_            
      rt
    radar cha# Close thevalues[0])  append(s. value    
       n metrics]or metric i) f 0etric,t(mata.ge = [expert_dalues       v   c[0]
  rt].ilome'] == expe_naperf['expertperf[df_df_= pert_data          ex   nique():
ame'].u'expert_n_perf[expert in df   for       
 ]
      ion'atisfact 'user_sy',bility', 'reliancieict_eff', 'cosspeed', ' ['accuracyetrics =   m   
        ()
  o.Figureg_radar = g fi    on
   arisional compimensti-d Mul
        #     ta)
   rmance_daerfoDataFrame(pf = pd.er    df_p    
   
     urn ret         ")
  ilable. data avae comparisonrmanco perfofo("N  st.in          _data:
rformancet pe      if no[])
  parison', mance_comt('perforta.getion_da utilizace_data = performan         
 
     )arison"Compce t Performan 🏆 Exper"###rkdown(    st.ma""
    "on.ompariserformance cexpert pRender ""    ":
    y]) -> None[str, Ana: Dictattion_dzaself, utiliomparison(xpert_cer_ef _rend
    de  
  h=True)ainer_widt use_conttrend,y_chart(fig_otl  st.pl                   )

   pert"rends by Ext Tosle="Daily C      tit       
   ert_name',olor='exp        c,
            y='cost'           'date',
        x=        f_daily,
      d      
     px.line(rend = ig_t          f
          e'])
    _daily['datdatetime(df pd.to_date'] =f_daily['         dcosts)
   daily_me(ra.DataFily = pd   df_da        ts']
 cosa['daily_daton_tis = utilizacost    daily_      _data:
  utilizationy_costs' in    if 'dail     er time
rend ov Cost t
        #   )
     dth=Truetainer_wie_con, uscyfig_efficienrt(_chat.plotly         s    )
          Queries'}
 l ': 'Totaotal_queries't                       ', 
$)ery (essful QuSucc: 'Cost per iciency'_eff={'costabels     l    ,
       "nalysisency Aficie="Cost Ef   titl       pe',
      rt_tyolor='expe  c           cost',
   otal_  size='t        
      ency',fficist_e       y='co
         queries',l_'tota        x=       ost,
 _c     df
           (x.scatter pfficiency =    fig_e  ']
      l_queriesccessfu df_cost['su'] /al_costst['tot = df_coiciency']effcost['cost_         df_
   ery) qusfulcesost per succiency (ct effi       # Cos:
     th col2  wi  
      
      ue)width=Trontainer_st, use_cg_co_chart(fiplotly  st.     )
     -45kangle=s_ticout(xaxie_lay_cost.updat        fig        )
    '}
    e': 'Expert 'expert_namost ($)',t': 'Total Ctal_cosbels={'to    la          pert",
  "Cost by Exe=titl       ,
         per_token'st_lor='co      co      cost',
       y='total_            name',
 xpert_        x='e        
 df_cost,          .bar(
     g_cost = px     fi      expert
  t perCos    #   1:
       col  with
             olumns(2)
 .c col2 = st    col1,    
    
    _data)aFrame(costost = pd.Dat  df_c
      n
        etur          rable.")
  a availo cost datnfo("N        st.i
     cost_data:     if not[])
   a', t('cost_data.geization_datta = utilst_da     co 
         ")
 isost Analys### 💰 C"wn(   st.markdo""
     ."lysis chartsnacost aer end""R "   
    -> None:) str, Any]data: Dict[tion_ utilizaelf,analysis(snder_cost_ def _re
   
    width=True)e_container_ries, usuet(fig_qy_charst.plotl     
         )    
      by Expert"tribution ="Query Disitle         t
       t_name',exper   names='            ueries',
 s='total_qalue     v      
     df,           pie(
     ries = px.queig_         fbution
    distri     # Query      th col2:
        wi
 
        dth=True)ainer_wi, use_contutilchart(fig_ st.plotly_         e=-45)
  ckanglt(xaxis_tiate_layouig_util.upd    f)
                   ert'}
 'Expt_name': 'experzation %', li: 'Ution_percent'ilizatibels={'ut          la",
      ion Rates Utilizaterttle="Exp    ti     
       pe',rt_tylor='expe      co       ',
   _percentlization   y='uti          ',
   t_name x='exper            df,
             
      x.bar(l = p_uti       figpert
     exy tilization b  # U       h col1:
        wit      
     s(2)
t.columnl2 = s1, co   col      
   )
    ts_datae(experram.DataF pdf =
        d        turn
        re   ")
 ilable.data avalization utit ("No expert.info         ss_data:
   t expert if no    , [])
   s'('expertgettion_data.zaata = utilierts_d    exp    
 ")
       rics Mettilizationn("### 📈 Uarkdow      st.m""
  metrics."zation li utiender expert    """R None:
    , Any]) ->: Dict[stration_dataelf, utilizetrics(s_mtioner_utiliza  def _rend  a)
    
on_dat(utilizatiomparisoner_expert_c._rend selfson
       nce comparirmaExpert perfo#                
_data)
 ontisis(utilizast_analycorender_f._
        selsalysi   # Cost an      
       tion_data)
iliza(utricsion_metzatliender_uti     self._r
   etricsn m# Utilizatio
        
        ysis")nalost A Cilization &pert Ut🤖 Exubheader("t.s      s
  """ew.overviation xpert utilizder e """Ren     None:
  -> , Any]) Dict[strdata: utilization_f, elew(servion_ovtiilizaer_ut   def rend
 ts")
    CharlizationrtUtiame__}.Expe(f"{__ngerogt_lger = geogf.lel       sig()
 yticsConf Analfig orconfig = conself.     :
    = None)nfiglyticsCoconfig: Ananit__(self,     def __i""
    
." chartsanalysisnd cost ation aizil""Expert utts:
    "onCharilizatixpertUt Eclassrue)


width=Ter_ntain, use_cochart(fig  st.plotly_
       Metrics")ystemal-Time Sxt="Re  title_te                     d=True, 
  enowlegght=600, sh(heiayoutig.update_l   f       
        )
      ol=2
, cw=2         ro')),
   brownct(color=', line=die'ritme='Disk W       na              ], 
 _write'disk, y=df['timestamp']er(x=df['   go.Scatt
         trace(   fig.add_ )
     l=2
       2, co  row=         ple')),
 'purdict(color=ne=lisk Read', ='Di  name               
     , read'] y=df['disk_tamp'],mesf['tiScatter(x=d go.       ace(
      fig.add_trI/O
        # Disk            
    )
 
      ow=2, col=1     r      
 orange')),lor='dict(coine=ork Out', lNetw='  name                    , 
etwork_out']=df['namp'], yestimdf['tr(x=teo.Scat           gce(
 ig.add_tra   f          )
=1
   ol   row=2, c
         ='green')),or(coline=dictk In', letwor   name='N                   _in'], 
df['networkestamp'], y=timr(x=df['go.Scatte          (
  add_traceig.
        fNetwork I/O #   
                    )
 l=2
coow=1,   r    ),
      blue')ct(color='=diliney %', 'Memor      name=          '], 
      y_percentf['memor'], y=dmestamp=df['tio.Scatter(x    g   e(
     acdd_trg.a    fi
    geemory Usa # M       
    )
     
       ol=1, c row=1       d')),
    t(color='ree=dicPU %', lin     name='C               cent'], 
  erdf['cpu_py=, timestamp'](x=df['ertto.Sca           g
 _trace(     fig.add
    CPU Usage 
        #       )
       e}]]
 ry_y": Fals {"seconday": False},condary_[{"se                   ],
lse}": Faondary_ylse}, {"secary_y": Faecondpecs=[[{"s     s),
       sk I/O'rk I/O', 'DiNetwoge', ' Usa'MemoryUsage', U s=('CPlot_title    subp    2,
    s=s=2, colrow        lots(
    = make_subp       fig ics chart
 e metrReal-tim   # 
     
        '])mestamp'time(df[d.to_dateti pestamp'] =tim   df['
     a)datics_etrDataFrame(m= pd. df   
         n
          retur.")
      s availableictime metrl-rea"No st.info(       data:
      metrics_    if not
    etrics', [])_time_mal'reget(h_data.ltta = heaics_da        metr      

  Metrics")eal-Time "### 📊 Rmarkdown(       st.s."""
 m metricime systeer real-t """Rend   one:
    , Any]) -> Ntr: Dict[sdatahealth_, selfrics(_time_met_render_real    def e)
    
ow_html=Tru unsafe_all """,         div>
        </             }</p>
 ', 'N/A')st_checkatus.get('la> {strongt Check:</stong>Las     <p><str              text}</p>
 ng> {status_atus:</strorong>St      <p><st          p>
    :.1%}</reealth_sco> {hh:</strongstrong>Healt><    <p                le()}</h4>
ponent.tit{comus_color} tat>{s         <h4           card">
metric-lass="iv c         <d"
       down(f""     st.mark      
                
     = "🔴"s_color       statu               else:
            
   "🟡color = "  status_              :
     > 0.6h_scorehealtlif          e       = "🟢"
 s_coloratu       st            .8:
 e > 0th_scoreal   if h        th
     ased on healding b  # Color co         
                    wn')
 ', 'unknoget('statusus._text = stat status           , 0.0)
    alth_score'us.get('hee = stat health_scor           % 3]:
    ith cols[i           w
  s()):ents.itemrate(compon in enumeatus)nent, stcompo    for i, ( 
    
       .columns(3) st     cols = grid
   t statusomponeneate c   # Cr   
     , {})
     omponents'et('ca.gat health_dponents =        com     
us")
   nt Stat# 🔧 Componeown("##  st.markd""
      s."statucomponent dividual in""Render       "-> None:
  Any]) tr, ct[s_data: Dilf, healthnt_status(secomponeer_def _rend      
 
          )', 0)
   tions_delta'connec.get(lth_data   delta=hea         s,
    tionve_connec acti             ons",
  nectiActive Con    "      ric(
       st.met         , 0)
  ns'e_connectiotivacget('h_data.lt= heannections e_coctiv        aol4:
         with c     
    )
   
           "inverse"_color= delta         ",
      :.2%}, 0.0)or_delta'ta.get('erralth_data=f"{he   del            ",
 ate:.2%}_r{error   f"            te",
  Ra "Error              t.metric(
          s)
   0.0or_rate', t('errta.ge= health_daor_rate err           
 3:   with col     
 
          )        "
 0.0):.1f}h, lta'e_deet('uptimlth_data.ga=f"+{healt    de            ",
ptime:.1f}h"{u         f     
  "Uptime",                st.metric(
            0.0)
e_hours', a.get('uptimhealth_date =        uptim
     h col2:        wit      

   )        
   _coloror=healthcollta_         de,
       :.1%}"a', 0.0)lthealth_de('a.get_dat"{healthelta=f d               
lth:.1%}",rall_hea"{ove   f         ,
     Health" "Overall           ic(
    trt.me        s
     "off"8 elseh > 0.ll_healtera ovmal" if "norolor =health_c          0)
  ', 0.erall_healtht('ov_data.ge = healtherall_health      ov
      with col1:
                umns(4)
ol4 = st.col col3, ccol2,    col1, "
    rs.""lth indicatooverall hea""Render 
        "None: -> str, Any])ct[h_data: Di healtf,tors(selalth_indicar_hef _rende   de
 a)
    lth_dats(heaetrictime_mer_real_self._rend       s
 metrical-time       # Re      
  th_data)
  us(healstatr_component_ self._rende
        statusntpone      # Com     
  data)
   th_(healtorsindicaer_health_self._rend
        rsindicatoatus  Health st       #      
 tor")
  lth Moni Heaystem"🏥 Sader(bhe      st.su""
   overview."m healthyste snderRe"       ""
 > None:r, Any]) -ct[stata: Dih_d, healtrview(selfr_health_ove def rende    
   ")
ealthMonitore__}.SystemHam_ner(f"{_get_logger = f.loggsel      nfig()
  csCo or Analyti= configfig self.con    ne):
     = NoicsConfigg: Analytlf, confise __init__(    def"
    
"".ng dashboardlth monitorim hea""Syste   "Monitor:
 SystemHealth
class 
=True)
idthtainer_w_conig, usey_chart(ft.plotl    s
    alytics")ciency An"System Effitext=le_        tit            ue, 
     egend=Tr0, showlut(height=60date_layofig.up              

         )ol=2
 w=2, c      ro     ,
 'purple'))ict(color=      line=d          ', 
      ll='tonextycore', fiance Sormame='Perf       n            , 
   core']_scermany=df['perfo], f['date'tter(x=d      go.Sca  (
    dd_trace    fig.a score
    ancerall perform # Ove              
    )
 col=1
     row=2,          y'),
   per Querost'], name='Cer_query['cost_p y=df],'date'.Bar(x=df[         go   race(
   fig.add_t    fficiency
  Cost e
        #             )
 =2
  ow=1, col           r
 een')),(color='grne=dict', liageemory Us name='M                   ge'], 
  y_usamor, y=df['mee']r(x=df['datScatte      go.ce(
      d_tra   fig.ad)
          2
   ol=ow=1, c         r),
   red')dict(color='ine=e', le='CPU Usag   nam                  
  pu_usage'],'c], y=df[te''daf[er(x=d.Scattgo        trace(
    g.add_      fition
  ilizaesource ut
        # R)
                l=1
   row=1, co,
         ='blue'))ct(colore=diiency', linry Efficame='Que          n        
     ncy'],y_efficie y=df['quer'],f['dateatter(x=d  go.Sc        trace(
  .add_fig
        fficiency erocessingery p # Qu      
   )
       
       alse}]]_y": Fecondarylse}, {"sy_y": Fa[{"secondar                 e}],
  _y": Falsary{"secondse}, : Faly_y"ndarsecocs=[[{"       spe  core'),
   rmance SPerfol Overalncy', 'ost Efficie   'C                     on', 
  tizarce Utiliy', 'Resouficienc Efocessingery Pr=('Questitl   subplot_
         ols=2,  rows=2, c         ots(
 _subpl make     fig =y chart
   efficiencti-metric  # Mul 
            date'])
  (df['o_datetimed.t = p  df['date'])
      y_dataencfficiDataFrame(e  df = pd.     
  n
            retur  e.")
     availabla ncy dat"No efficie st.info(
           data:iency_ efficif not   , [])
     ency_data'ficiet('efta.gformance_da = perciency_data     effi    
   s")
    ndency Tre# 📊 Efficiarkdown("##.m      st""
  trends."improvement iciency der eff"Ren  "":
      ny]) -> None Dict[str, Adata:erformance_s(self, pendiciency_trrender_eff def _
    
   th=True)tainer_wid, use_conimelineart(fig_tchlotly_st.p          
             )
       
      ency}s"lat {target_f"Target:ation_text=       annot    
     range",color="one_li            h",
    _dash="das line           cy,
    arget_laten    y=t           e(
 add_hlinmeline.ig_ti         fents
   iremequfrom rarget second t0  # 5  = 5.t_latency targe        line
   Add target        #              
   )
    
         '}mp': 'Timeesta', 'tim (seconds)Latencyency': 'bels={'lat       la       ",
   Over Time"Latency      title=
          ency',y='lat             
   stamp',    x='time         df,
           (
        px.linetimeline =   fig_         '])
 ['timestampetime(df.to_datmp'] = pd['timesta df       ns:
    df.columestamp' in im    if 't  e
  ver timLatency o        #  
e)
       _width=Tru_containerseg_expert, uy_chart(fi.plotl         st)
       45ickangle=-axis_tayout(xdate_lxpert.up    fig_e                  )

          rt"peEx by Latencytitle="                 ncy',
        y='late          ',
     expert      x='      ,
                df         (
   px.boxrt = g_expe      fi   ns:
       in df.colum'expert'  if 
           y experty b    # Latenc        ith col2:
   w     
     
   h=True)tainer_widtst, use_con_diig(ftly_chartst.plo                  )
 s"
     f}():.2eany'].mlatenc['"Mean: {dfon_text=fotati   ann      ,
       r="red"   line_colo             h",
ash="das    line_d            .mean(),
y']['latencdf          x=ine(
      _dist.add_vl    fig   )
       
          quency'}'Fre': )', 'countnds (seco: 'Latencyncy'={'latels        labe
        ution",y Distribatenc"Lle=        tit       ins=30,
    def _render_latency_analysis(self, performance_data: Dict[str, Any]) -> None:
        """Render latency analysis charts."""
        st.markdown("### ⏱️ Latency Analysis")
        
        latency_data = performance_data.get('latency_data', [])
        if not latency_data:
            st.info("No latency data available.")
            return
            
        # Create DataFrame for analysis
        df = pd.DataFrame(latency_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Latency distribution histogram
            fig_dist = px.histogram(
                df,
                x='latency',
                nbins=20,
                title="Latency Distribution",
                labels={'latency': 'Latency (seconds)', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Performance trends
            self._render_efficiency_trends(performance_data)
    
    def _render_performance_metrics(self, performance_data: Dict[str, Any]) -> None:
        """Render performance metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            throughput = performance_data.get('throughput', 0.0)
            throughput_delta = performance_data.get('throughput_delta', 0.0)
            st.metric(
                "Throughput",
                f"{throughput:.1f} q/s",
                delta=f"{throughput_delta:.1f}",
                help="Queries per second"
            )
        
        with col2:
            success_rate = performance_data.get('success_rate', 0.0)
            success_delta = performance_data.get('success_delta', 0.0)
            st.metric(
                "Success Rate",
                f"{success_rate:.1%}",
                delta=f"{success_delta:.1%}"
            )
        
        with col3:
            efficiency_improvement = performance_data.get('efficiency_improvement', 0.0)
            st.metric(
                "Efficiency Gain",
                f"{efficiency_improvement:.1%}",
                delta=f"{performance_data.get('efficiency_delta', 0.0):.1%}"
            )
        
        with col4:
            avg_latency = performance_data.get('avg_latency', 0.0)
            latency_delta = performance_data.get('latency_delta', 0.0)
            st.metric(
                "Avg Latency",
                f"{avg_latency:.2f}s",
                delta=f"{latency_delta:.2f}s"
            )