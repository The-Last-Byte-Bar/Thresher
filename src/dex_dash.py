from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
from datetime import datetime
import numpy as np
from pathlib import Path
import dash_bootstrap_components as dbc
from metrics import RLMetricsCalculator, TokenMetrics

class EnhancedDEXDashboard:
    def __init__(self, data_dir: str = "data"):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.data_dir = Path(data_dir)
        self.swaps_data = None
        self.tokens = []
        self.token_data = {}
        self.rl_metrics = RLMetricsCalculator()
        
        self.setup_layout()
        self.setup_callbacks()
        
    def load_data(self, filename: str = None):
        """Load and process swap data"""
        if not filename:
            json_files = list(self.data_dir.glob("swaps_*.json"))
            if not json_files:
                raise FileNotFoundError("No swap data files found")
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        else:
            latest_file = self.data_dir / filename
            
        with open(latest_file) as f:
            self.swaps_data = json.load(f)
            
        # Process data by token and update RL metrics
        self.token_data = {}
        for swap in self.swaps_data['swaps']:
            token = swap['output_token_name'] if swap['input_token'] == 'ERG' else swap['input_token_name']
            if token == 'ERG':
                continue
                
            if token not in self.token_data:
                self.token_data[token] = []
            self.token_data[token].append(swap)
            
            # Update RL metrics
            self.rl_metrics.update_with_swap(swap)
            
        self.tokens = list(self.token_data.keys())
        
    def calculate_metrics(self, token_swaps):
        """Calculate trading metrics from swaps"""
        df = pd.DataFrame(token_swaps)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['is_buy'] = df['input_token'] == 'ERG'
        
        # Calculate moving averages
        df['MA5'] = df['price'].rolling(window=5).mean()
        df['MA20'] = df['price'].rolling(window=20).mean()
        
        # Calculate volume in ERG
        df['volume_erg'] = np.where(
            df['is_buy'],
            df['input_amount_adjusted'],
            df['output_amount_adjusted']
        )
        
        # Calculate buy pressure
        df['buy_pressure'] = df['is_buy'].rolling(20).mean()
        
        return df

    def create_token_figures(self, token):
        if token not in self.token_data:
            return go.Figure()
            
        df = self.calculate_metrics(self.token_data[token])
        metrics = self.rl_metrics.current_metrics.get(token)
        
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           row_heights=[0.5, 0.25, 0.25],
                           subplot_titles=(f'{token} Price and MAs', 'Volume (ERG)', 'Buy Pressure'))
        
        # Price and MAs
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['price'],
                      name='Price', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['MA5'],
                      name='MA5', line=dict(color='orange')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['MA20'],
                      name='MA20', line=dict(color='red')),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['volume_erg'],
                  name='Volume (ERG)'),
            row=2, col=1
        )
        
        # Buy Pressure
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['buy_pressure'],
                      name='Buy Pressure', line=dict(color='green')),
            row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title=f'{token} Trading Activity and Metrics'
        )
        
        return fig
        
    def create_metrics_cards(self, token):
        if token not in self.rl_metrics.current_metrics:
            return []
            
        metrics = self.rl_metrics.current_metrics[token]
        signals = self.rl_metrics.get_potential_signals(token)
        
        cards = [
            dbc.Card([
                dbc.CardHeader("Price Metrics"),
                dbc.CardBody([
                    html.P(f"Current Price: {metrics.current_price:.6f}"),
                    html.P(f"MA5: {metrics.price_ma_5:.6f}"),
                    html.P(f"MA20: {metrics.price_ma_20:.6f}"),
                    html.P(f"Volatility: {metrics.price_volatility:.4f}")
                ])
            ]),
            
            dbc.Card([
                dbc.CardHeader("Volume & Liquidity"),
                dbc.CardBody([
                    html.P(f"Volume MA5: {metrics.volume_ma_5:.2f} ERG"),
                    html.P(f"ERG Liquidity: {metrics.erg_liquidity:.2f}"),
                    html.P(f"Token Liquidity: {metrics.token_liquidity:.2f}")
                ])
            ]),
            
            dbc.Card([
                dbc.CardHeader("Trading Signals"),
                dbc.CardBody([
                    html.P(f"MA Crossover: {signals['ma_crossover']:.4f}"),
                    html.P(f"Price Trend: {signals['price_trend']:.4f}"),
                    html.P(f"Buy Pressure: {signals['buy_pressure']:.4f}"),
                    html.P(f"Volatility Signal: {signals['volatility_signal']:.4f}")
                ])
            ])
        ]
        
        return dbc.Row([dbc.Col(card, width=4) for card in cards])
        
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Enhanced DEX Trading Dashboard"),
                    html.Hr(),
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='token-selector',
                        options=[],
                        value=None,
                        placeholder="Select a token"
                    ),
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Div(id='metrics-cards'),
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='token-chart'),
                ])
            ]),
            
        ], fluid=True)
        
    def setup_callbacks(self):
        @self.app.callback(
            [Output('token-selector', 'options'),
             Output('token-selector', 'value')],
            Input('token-selector', 'id')
        )
        def update_token_dropdown(_):
            options = [{'label': token, 'value': token} for token in self.tokens]
            initial_value = self.tokens[0] if self.tokens else None
            return options, initial_value
            
        @self.app.callback(
            [Output('token-chart', 'figure'),
             Output('metrics-cards', 'children')],
            Input('token-selector', 'value')
        )
        def update_charts(token):
            if not token:
                return go.Figure(), []
                
            fig = self.create_token_figures(token)
            metrics_cards = self.create_metrics_cards(token)
            
            return fig, metrics_cards
            
    def run_server(self, debug=True, port=8050):
        self.load_data()  # Load data before starting server
        self.app.run_server(debug=debug, port=port)

if __name__ == '__main__':
    dashboard = EnhancedDEXDashboard()
    dashboard.run_server()