# Arquitetura de Computadores III — Trabalho 1

## Sobre o Projeto

Este trabalho tem como intuito aprofundar o estudo da **hierarquia de memória** em arquiteturas modernas de processadores, utilizando simuladores arquiteturais de nível de pesquisa. Em vez de se limitar a ferramentas educacionais simplificadas, o trabalho busca desenvolver uma visão de **Arquitetura de Computadores como ciência experimental**, aproximando os alunos de metodologias empregadas em pesquisa acadêmica.

Por meio de experimentos controlados em um simulador completo de sistema, é possível observar, medir e analisar fenômenos que normalmente ficam ocultos durante a execução de programas em hardware real, como o comportamento interno de caches, os padrões de acesso à memória e o impacto de diferentes decisões de projeto no desempenho final.

---

## Objetivos

Propor cenários experimentais capazes de validar conceitos fundamentais relacionados a:

- **Localidade temporal e espacial** — verificar como padrões de acesso sequenciais ou repetitivos afetam o desempenho em função do reaproveitamento de dados já presentes na cache.
- **Comportamento de caches** — analisar métricas como taxa de acertos (*hit rate*), taxa de faltas (*miss rate*) e o efeito de diferentes políticas de substituição e tamanhos de bloco.
- **Interação CPU–memória** — entender os gargalos criados pela latência e largura de banda da memória principal e como a hierarquia de cache os atenua.
- **Impacto de decisões arquiteturais no desempenho** — comparar configurações distintas (tamanho de cache, associatividade, número de níveis, etc.) e avaliar seus efeitos sobre o tempo de execução e o consumo de energia.

---

## Simulador Utilizado: gem5

O simulador escolhido para este trabalho é o **[gem5](https://www.gem5.org/)**, um simulador arquitetural de código aberto amplamente utilizado tanto na indústria quanto na academia.

### Por que o gem5?

O gem5 se destaca entre os simuladores disponíveis por uma série de razões:

| Característica | gem5 | Sniper |
|---|---|---|
| **Modelo de simulação** | Ciclo a ciclo (*cycle-accurate*) ou funcional | Baseado em intervalos (*interval simulation*) |
| **Fidelidade** | Alta — modela o pipeline completo, caches, interconexões e dispositivos | Média — otimizado para velocidade, com menor detalhamento microarquitetural |
| **ISAs suportadas** | x86, ARM, RISC-V, MIPS, SPARC, Power e outras | Principalmente x86 |
| **Modelo de memória** | Sistema de memória altamente configurável (Ruby/Classic) | Menos flexível para customização |
| **Comunidade e documentação** | Extensa; mantido ativamente por universidades e empresas | Menor comunidade; desenvolvimento menos ativo |
| **Uso em pesquisa** | Referência em conferências como ISCA, MICRO e HPCA | Utilizado principalmente para simulações de alto nível |

### Vantagens do gem5 em relação ao Sniper

1. **Precisão microarquitetural**: o gem5 permite simular o pipeline do processador ciclo a ciclo, fornecendo dados detalhados sobre stalls, dependências de dados e comportamento de unidades funcionais — algo que o Sniper simplifica em favor da velocidade.
2. **Flexibilidade de configuração**: é possível definir livremente topologias de cache (L1, L2, L3), políticas de coerência, protocolos de barramento e modelos de memória DRAM sem grandes restrições.
3. **Suporte a múltiplas ISAs**: enquanto o Sniper foca quase exclusivamente em x86, o gem5 suporta ARM, RISC-V e outras arquiteturas, permitindo estudos comparativos entre diferentes famílias de processadores.
4. **Integração com ferramentas de pesquisa**: o gem5 se integra com McPAT (estimativa de potência/área), DRAMSim e outros frameworks, possibilitando análises mais completas.
5. **Código aberto e extensível**: escrito em Python e C++, o gem5 permite modificações profundas na microarquitetura simulada, algo essencial para pesquisa experimental.

---

## Estrutura do Repositório

```
.
├── README.md       # Este arquivo
└── LICENSE
```

> Os scripts de simulação, workloads e resultados experimentais serão adicionados ao longo do desenvolvimento do trabalho.

---

## Requisitos

- Python 3.x
- gem5 (instruções de instalação disponíveis em [gem5.org](https://www.gem5.org/documentation/general_docs/building))
- Compilador C++17 ou superior (g++ ou clang++)
- SCons (sistema de build utilizado pelo gem5)

---

## Como Executar

> *Em breve — as instruções de execução serão detalhadas conforme os experimentos forem desenvolvidos.*

---

## Equipe

Trabalho desenvolvido por alunos da disciplina **Arquitetura de Computadores III**.
