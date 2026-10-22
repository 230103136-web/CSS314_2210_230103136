import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

public class task6p {
    static int threads = 8;
    static int count = 1000000;
    static int total = threads * count;

    static int counter = 0;
    static AtomicInteger atomic = new AtomicInteger(0);
    static ReentrantLock lock = new ReentrantLock();

    static void runThreads(boolean useCAS) throws Exception {
        Thread[] jobs = new Thread[threads];

        for (int i = 0; i < threads; i++) {
            jobs[i] = new Thread(() -> {
                for (int j = 0; j < count; j++) {
                    if (useCAS) {
                        int old;
                        do {
                            old = atomic.get();
                        } while (!atomic.compareAndSet(old, old + 1));
                    } else {
                        lock.lock();
                        try {
                            counter++;
                        } finally {
                            lock.unlock();
                        }
                    }
                }
            });
            jobs[i].start();
        }

        for (Thread t : jobs) {
            t.join();
        }
    }

    public static void main(String[] args) throws Exception {
        long start = System.nanoTime();
        runThreads(false);
        double t1 = (System.nanoTime() - start) / 1e6;

        start = System.nanoTime();
        runThreads(true);
        double t2 = (System.nanoTime() - start) / 1e6;

        double ops1 = total / (t1 / 1000);
        double ops2 = total / (t2 / 1000);

        System.out.println("ID: 230103136");
        System.out.println("Threads: " + threads);
        System.out.printf("Mutex: %.2f ms%n", t1);
        System.out.printf("Mutex ops: %.2f%n", ops1);
        System.out.printf("CAS: %.2f ms%n", t2);
        System.out.printf("CAS ops: %.2f%n", ops2);
        System.out.printf("Ratio: %.2f%n", ops2 / ops1);
        System.out.println("Match: " +
            (counter == total && atomic.get() == total));
    }
}
