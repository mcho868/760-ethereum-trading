import asyncio
import logging
import signal
import sys
import json
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import numpy as np

# Import our components
from model_loader import ModelLoader
from data_processor import DataProcessor

class TradingSystem:
    """
    Simple DRL trading system that runs a trained agent in real-time
    Feeds market data to the agent and monitors its actions and rewards
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.is_running = False
        
        # Components
        self.model_loader: Optional[ModelLoader] = None
        self.data_processor: Optional[DataProcessor] = None
        
        # Trading state
        self.current_position = 0.0  # Current position in ETH
        self.cash_balance = config.get('initial_capital', 10000.0)
        self.portfolio_value = self.cash_balance
        self.trades_count = 0
        
        # Performance tracking
        self.actions_history = []
        self.rewards_history = []
        self.portfolio_history = []
        self.start_time = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _initialize_components(self):
        """Initialize model loader and data processor"""
        try:
            # Load model
            model_path = self.config.get('model_path')
            scaler_path = self.config.get('scaler_path')
            
            if not model_path:
                raise ValueError("model_path required in config")
                
            self.model_loader = ModelLoader(model_path, scaler_path)
            self.logger.info(f"Model loaded: {self.model_loader.get_model_info()}")
            
            # Initialize data processor
            symbol = self.config.get('symbol', 'ethusdt')
            self.data_processor = DataProcessor(symbol=symbol)
            
            # Register callback for new data
            self.data_processor.register_callback(self._on_new_data)
            
            self.logger.info("All components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
        
    async def start(self):
        """Start the trading system"""
        self.logger.info("Starting DRL Trading System")
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            # Start data processor
            data_task = asyncio.create_task(self.data_processor.start())
            
            # Start monitoring loop
            monitor_task = asyncio.create_task(self._monitor_loop())
            
            # Wait for tasks
            await asyncio.gather(data_task, monitor_task, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error in trading system: {e}")
        finally:
            await self.stop()
            
    async def stop(self):
        """Stop the trading system"""
        self.logger.info("Stopping trading system...")
        self.is_running = False
        
        if self.data_processor:
            await self.data_processor.stop()
            
        # Generate final report
        self._generate_report()
        self.logger.info("Trading system stopped")
        
    async def _on_new_data(self, market_state: Dict):
        """Callback when new market data arrives"""
        try:
            if not self.data_processor.is_ready():
                return
                
            # Get feature vector
            features = market_state.get('feature_vector')
            if features is None:
                return
                
            # Get agent action
            action = self.model_loader.predict(features, deterministic=True)
            
            # Execute action (simulate trading)
            reward = self._execute_action(action, market_state['price'])
            
            # Store history
            self.actions_history.append(action)
            self.rewards_history.append(reward)
            self.portfolio_history.append(self.portfolio_value)
            
            # Log periodically
            if len(self.actions_history) % 10 == 0:
                self._log_status(action, market_state['price'], reward)
                
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            
    def _execute_action(self, action: float, current_price: float) -> float:
        """
        Execute trading action and calculate reward
        This simulates the trading environment's step function
        """
        # Calculate target position based on action
        max_position_value = self.portfolio_value * 0.95  # 95% max exposure
        target_position = action * (max_position_value / current_price)
        
        # Calculate position change
        position_change = target_position - self.current_position
        
        # Only trade if significant change
        min_trade_size = 0.01  # Minimum 0.01 ETH
        if abs(position_change) < min_trade_size:
            return 0.0  # No reward for no action
            
        # Calculate trade value and fees
        trade_value = abs(position_change) * current_price
        fee = trade_value * 0.001  # 0.1% trading fee
        
        # Update position and cash
        old_position = self.current_position
        self.current_position = target_position
        
        if position_change > 0:  # Buying
            self.cash_balance -= (trade_value + fee)
        else:  # Selling
            self.cash_balance += (trade_value - fee)
            
        # Update portfolio value
        self.portfolio_value = self.cash_balance + (self.current_position * current_price)
        
        # Calculate reward (simple P&L based)
        if len(self.portfolio_history) > 0:
            portfolio_change = self.portfolio_value - self.portfolio_history[-1]
            reward = portfolio_change / self.portfolio_history[-1]  # Percentage change
        else:
            reward = 0.0
            
        # Increment trade count
        if abs(position_change) >= min_trade_size:
            self.trades_count += 1
            
        return reward
        
    async def _monitor_loop(self):
        """Monitor system performance and log status"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                self._log_system_status()
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                
    def _log_status(self, action: float, price: float, reward: float):
        """Log current trading status"""
        self.logger.info(
            f"Action: {action:+.4f} | Price: ${price:.2f} | Position: {self.current_position:.4f} ETH | "
            f"Portfolio: ${self.portfolio_value:.2f} | Reward: {reward:+.6f}"
        )
        
    def _log_system_status(self):
        """Log system performance summary"""
        if not self.actions_history:
            return
            
        runtime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        total_return = ((self.portfolio_value - self.config['initial_capital']) / 
                       self.config['initial_capital']) * 100
        
        avg_reward = np.mean(self.rewards_history) if self.rewards_history else 0
        
        self.logger.info(
            f"=== SYSTEM STATUS === Runtime: {runtime_hours:.1f}h | "
            f"Actions: {len(self.actions_history)} | Trades: {self.trades_count} | "
            f"Return: {total_return:+.2f}% | Avg Reward: {avg_reward:+.6f}"
        )
        
    def _generate_report(self):
        """Generate final trading report"""
        if not self.start_time:
            return
            
        runtime = datetime.now() - self.start_time
        initial_capital = self.config['initial_capital']
        total_return = ((self.portfolio_value - initial_capital) / initial_capital) * 100
        
        report = {
            'session_duration': str(runtime),
            'initial_capital': initial_capital,
            'final_portfolio_value': self.portfolio_value,
            'total_return_pct': total_return,
            'total_actions': len(self.actions_history),
            'total_trades': self.trades_count,
            'final_position': self.current_position,
            'cash_balance': self.cash_balance,
            'avg_reward': np.mean(self.rewards_history) if self.rewards_history else 0,
            'total_reward': np.sum(self.rewards_history) if self.rewards_history else 0,
            'max_portfolio_value': max(self.portfolio_history) if self.portfolio_history else initial_capital,
            'min_portfolio_value': min(self.portfolio_history) if self.portfolio_history else initial_capital
        }
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"trading_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.logger.info(f"Trading report saved to {report_file}")
        self.logger.info(f"Final Report: {json.dumps(report, indent=2, default=str)}")

def load_config() -> Dict:
    """Load configuration"""
    config_file = Path("trading_config.json")
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        # Default config
        default_config = {
            "model_path": "/Users/choemanseung/4th year/760/760-ethereum-trading/Test_drl_training/a2c_ultra_conservative_0001.zip",
            "scaler_path": None,  # Add if you have a scaler file
            "symbol": "ethusdt", 
            "initial_capital": 10000.0
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
            
        return default_config

async def main():
    """Main execution"""
    try:
        config = load_config()
        system = TradingSystem(config)
        await system.start()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("DRL Trading System Starting...")
    print("Press Ctrl+C to stop")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
    except Exception as e:
        print(f"System error: {e}")
        sys.exit(1)