# Test-time Computeのレポジトリ
ライブラリllm-reasonersをベースとして、MBTIの思考トレースを模倣するtest-time compute手法を実験するレポジトリ

# セットアップ
1. 必要なライブラリをインストールします。uvでなくても良いです。
```bash
uv venv
./.venv/bin/activate
uv pip install .
```
2. .venv/lib/python3.11/site-packages/reasoners/lm/hf_model.pyをchanges/hf_model.pyに置き換えてください。
3. main.pyを実行して推論

# TODO
- llm-jp/llm-jp-3-13b-instruct3を指定したときに出るエラーの解消
    - 今の所HFModelを使っており、self.model.generateで"token_type_ids"が渡されてエラーが生じる模様
    - HFModel側を書き換えるか、versionを落とすか、他のModelを使うか検討
    - 現状はself.tokeinzerでreturn_token_type_ids=Falseを指定して対処
- 推論時間が異様に長い
    - バッチ推論していないから遅い
        - max_batch_sizeを大きくしたらバッチ
        - どちらにせよ、単一のinputsを拡張してバッチするしかできない
        - 探索回数に比例して、推論が増える
    - 単純にHFの推論の最適化がされていない
        - 実施ずみ
            - flashattention2
        - 検討事項
            - quantize vs (torch.compile, bfloat16)
            - HFから乗り換える
            - static kv cacheとtorch.compileの併用
- 目安時間
    - quantize=fp4だと40s/it
    - bfloat16, torch.compile, static kv cacheだとよくわからない挙動
        - 生成が徐々に加速している, 30s/it→16s/it→まだ早くなりそう
        - 速さと品質からこちらの方が良さそう