# Preflight Tests

验证长任务预测、GPU 利用率门、compute-app 归属、外部任务拒绝、隔离短训练预测，以及周期运行的配置与资源门控。外部 compute 进程存在时，测试要求在创建校准模型目录前写出 `failed_preflight`。
