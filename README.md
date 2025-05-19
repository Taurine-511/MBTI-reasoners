# Test-time Computeのレポジトリ
ライブラリllm-reasonersをベースとして、MBTIの思考トレースを模倣するtest-time compute手法を実験するレポジトリ


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