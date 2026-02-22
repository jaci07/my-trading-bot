# 🚀 Advanced Triple-Engine AI Trader (MT5)

This repository houses a high-performance **algorithmic trading engine** designed for MetaTrader 5. It seamlessly integrates institutional volume structure analysis with a three-layered Machine Learning approach (Random Forest) to evaluate trade probabilities in real-time.

---

### 🧠 Core Features

* **Triple-Timeframe AI Consensus**
  The engine utilizes a hierarchical decision logic across three distinct timeframes:
  * **(Not implemented yet!)M15 (Macro):** Establishes the institutional trend and structural market bias.
  * **M5 (Execution):** Identifies specific volume setups such as VAH/VAL breakouts and rejections.
  * **M1 (Precision):** Acts as a high-speed volatility filter to pinpoint exact entry timing.

* **Volume Profile Engine**
  Validates price movements by analyzing actual volume distribution:
  * Calculation of the **Value Area (VAH/VAL)** and the **Point of Control (POC)**.
  * Identification of **Low Volume Areas (LVAs)** for strategic "protected" Stop-Loss placement.

* **Shadow Trading Memory**
  The engine continuously spawns **virtual trade variants** in the background. These "Shadow Trades" analyze various SL/TP scenarios and optimize the AI models without risking live capital.

* **Smart Trade Management**
  * **Automated Break-Even:** Protects capital by moving the SL once partial profit targets are met.
  * **ATR-Based Trailing SL:** Dynamic risk management that adjusts to current market volatility.
  * **Night Guard:** Automatically secures or closes positions before high-spread rollover hours.

* **Remote Control**
  Fully integrated with a **Discord Bot** for real-time monitoring, status reports, and remote account switching.

---

### 🛠️ AI Training Protocol (CRITICAL)

The AI models are broker-specific. You **must** train them yourself to account for the unique spreads and liquidity profiles of your specific trading account.

#### Step-by-Step Instructions:

1. **Configuration:**
   * Select your desired symbols in `settings.py`.
   * Ensure MetaTrader 5 has fully downloaded the history for M1, M5, and M15 for all selected assets.

2. **Launch Training:**
   * Run `trainer.py`.
   * The process can take anywhere from a few minutes to several hours, depending on the number of assets and your hardware.

3. **Resuming Progress:**
   * If you need to stop the training, remove the already completed symbols from the list in `settings.py` and restart the script later.
   * Completed models are saved automatically in the `ai_models/` folder as `.pkl` files.

4. **Ready for Deployment:**
   * Once the models are generated, `main.py` will automatically recognize, load, and use them for live market scanning.

---

> **Note:** A model is considered stable if the "True Test Accuracy" shown in the logs exceeds 52% (assuming a Risk-Reward Ratio of 1:1.5 or higher).
