def apply_thread_limit(cfg: dict) -> None:
    """PyTorch/OpenCV thread sayısını config general.torch_threads'e sınırlar (0 = sınırsız).

    Ultralytics hem import'ta hem her modelin İLK tahmininde torch'u 8 thread'e ayarlıyor; bu yüzden detektörler
    bunu her tahminden SONRA çağırır (değişmemişse hiçbir şey yapmaz, maliyeti yok). Ölçüm (12 çekirdekli laptop,
    tam hat): 8 thread 21 FPS / 8.2 çekirdek, 4 thread 16 FPS / 4.6 çekirdek. Sunumda laptop ısınıp yavaşlamasın.
    """
    n = int(cfg["general"].get("torch_threads", 0))
    if n <= 0:
        return
    import torch

    if torch.get_num_threads() != n:
        import cv2

        torch.set_num_threads(n)
        cv2.setNumThreads(n)
