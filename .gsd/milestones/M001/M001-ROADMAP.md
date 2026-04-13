# M001: Production-Grade Sales Agent V2

## Vision
升级为生产级汽车出口销售AI Agent：智能价格检索+三步自反思工作流+人工介入机制，达到可对接企业知识库和销售系统的水准。

## Slice Overview
| ID | Slice | Risk | Depends | Done | After this |
|----|-------|------|---------|------|------------|
| S01 | UI Bug修复 | low | — | ⬜ | sidebar正常开合，WhatsApp tab输入框正常显示 |
| S02 | 增强RAG价格检索模块 | medium | S01 | ⬜ | 问BYD Seal返回带ID和置信度的报价，问错误车型触发人工介入 |
| S03 | 三步自反思工作流 | high | S02 | ⬜ | 回复价格时sidebar显示三步检查结果和反思日志 |
| S04 | 人工介入热键+权限系统 | high | S03 | ⬜ | 点击介入按钮暂停回复，编辑后恢复，admin可看所有会话 |
| S05 | 对接说明文档 | low | S04 | ⬜ | README中有清晰的对接步骤 |
