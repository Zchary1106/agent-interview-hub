# Java 核心基础：语言、集合、异常、JVM 与并发

> 对应学习计划「Java 后端基础」阶段。目标：集合选型、并发工具、JVM 内存与 GC 都能用面试语言讲清楚。

---

## 一、核心知识要点

### 1. 集合框架

| 场景 | 推荐容器 | 关键点 |
|------|---------|--------|
| 有序可重复访问 | ArrayList | 默认容量 10，扩容 1.5 倍，随机访问 O(1) |
| 频繁头尾增删 | LinkedList | 双链表，不适合随机访问 |
| 快速查找 | HashMap | 数组 + 链表/红黑树；负载因子 0.75；树化阈值 8 |
| 有序 Map | TreeMap / LinkedHashMap | 红黑树 / 插入序或访问序 |
| 并发读多写少 | CopyOnWriteArrayList / ConcurrentHashMap | 写时复制 / 分段锁（JDK8 为 CAS + synchronized） |

### 2. 异常体系

- 根类是 `Throwable`，分为 `Error`（不可恢复，如 OOM）与 `Exception`。
- `Exception` 分受检异常（checked，必须处理）与运行时异常（unchecked，如 NPE）。
- 使用原则：异常用于异常路径，不要用异常控制正常流程；`finally` 中释放资源，或用 try-with-resources。

### 3. JVM 内存与 GC

- 运行时数据区：堆、虚拟机栈、本地方法栈、方法区（元空间）、程序计数器。
- 判断可回收：可达性分析（GC Roots：栈帧局部变量、静态变量、常量、JNI 引用）。
- 分代收集：新生代（Eden + 两个 Survivor，复制算法）、老年代（标记-整理）；常用收集器 G1（Region + 停顿预测）。
- 调优入口：`-Xms/-Xmx`、`-XX:MaxMetaspaceSize`、GC 日志与 heap dump 分析。

### 4. 并发基础

- 线程创建方式：`Thread`、`Runnable`、`Callable + FutureTask`、线程池；实际工程用线程池。
- 线程池核心参数：corePoolSize、maximumPoolSize、workQueue、keepAliveTime、拒绝策略。
- 关键工具：`synchronized` / `ReentrantLock`、`CountDownLatch` / `CyclicBarrier`、`Semaphore`、`CompletableFuture`。
- 原子性/可见性/有序性：volatile 保证可见性与有序性（禁止重排），原子性靠 CAS 或锁。

## 二、常见误区

- 误区：HashMap 线程安全。纠正：HashMap 并发下可能丢数据或环形链表，应使用 ConcurrentHashMap。
- 误区：内存越大 GC 越频繁性能越好。纠正：堆过大导致 Full GC 停顿更长，需要结合对象生命周期设置分代大小。
- 误区：多线程一定更快。纠正：线程切换与锁竞争有开销，CPU 密集任务线程数建议 ≈ 核数。

## 三、高频面试题（附参考答案）

1. ArrayList 和 LinkedList 的区别？ArrayList 扩容机制？

> ArrayList 基于动态数组，随机访问 O(1)，尾插 O(1) 均摊；扩容时创建 1.5 倍新数组并 System.arraycopy。LinkedList 基于双链表，头尾增删 O(1)，随机访问 O(n)。频繁中间插入选 LinkedList，读多写少选 ArrayList。

2. HashMap 底层结构与并发问题？

> JDK8 为数组 + 链表 + 红黑树（链表长度 >8 且数组 ≥64 时树化）；put 流程：hash → 定位桶 → 尾插 → 扩容。JDK7 头插法在并发扩容时可能形成环形链表导致死循环；并发场景用 ConcurrentHashMap（JDK8 用 CAS + synchronized 锁桶）。

3. 线程池核心参数与提交流程？

> corePoolSize 核心常驻线程；任务提交后先入队列（workQueue），队列满再创建线程到 maximumPoolSize，仍满则走拒绝策略（Abort/CallerRuns/Discard/DiscardOldest）。keepAliveTime 决定非核心线程空闲回收时间。

4. 如何判断对象可回收？GC Roots 有哪些？

> 可达性分析：从 GC Roots 出发不可达的对象可回收。GC Roots 包括虚拟机栈中的局部变量、静态字段引用、常量引用、JNI 引用等。对象还可通过 finalize 自救一次（不推荐依赖）。

5. volatile 与 synchronized 的区别？

> volatile 只保证可见性与有序性，不保证原子性；synchronized 同时保证原子性、可见性与有序性，但有锁开销。复合操作（如 i++）需要锁或原子类（AtomicInteger）。

6. 线程池如何选择拒绝策略？

> 可丢弃任务用 DiscardPolicy/DiscardOldestPolicy；不能丢的用 CallerRunsPolicy（提交线程自己执行）或自定义告警 + 降级；生产上建议结合监控对队列长度和拒绝次数告警。

## 四、动手练习与掌握标准

- 练习：手写一个固定大小线程池 + 拒绝策略；用 jstack 观察死锁；用 jmap/jstat 分析一次 OOM 的堆。
- 掌握标准：能在白板上画出 HashMap put 流程；能解释"为什么 ConcurrentHashMap 读不加锁"；能说出一次线上 Full GC 排查步骤。
