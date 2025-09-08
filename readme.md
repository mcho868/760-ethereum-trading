2025/9/4
Currently, we have a complete workflow for processing Reddit historical social media data, which is stored in the Reddit_work_flow folder. Since this is historical data, our data source is not the official Reddit API (which has limitations for retrieving historical data), but instead comes from compressed historical data packages (zst format) downloaded from a website called Academic Torrents. Therefore, the entire Reddit data processing workflow starts with filtering Ethereum-related trading data from the zst packages.

According to Brendan’s suggestion, we can try not only using VADER for text scoring but also applying the five small language models mentioned in the literature. This means the next step will be to modify the relevant code in the CleanAndScoring file.

In addition, regarding DRL training, we have uploaded a Jupyter Notebook file in the drl folder. This notebook is used to test training functions and feedback, and it is divided into four parts: data preprocessing, environment setup, DRL learning, and out-of-sample testing & visualization. The current code can run end-to-end, but many parameters and rules (such as normalization of volume-price data, training rules, and environment settings) still need to be adjusted and added later.

Jimmy & Yolanda
