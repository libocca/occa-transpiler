extern "C" void hello_kern() {
  for (int i = 0; i < 10; ++i) {
    int shm[10];
    for (int j = 0; j < 10; ++j) {
      shm[j] = j;
    }
    for (int j = 0; j < 10; ++j) {
      shm[j] = j;
    }
    for (int j = 0; j < 10; ++j) {
      shm[j] = j;
    }
    for (int j = 0; j < 10; ++j) {
      shm[j] = j;
    }
  }
}

extern "C" void priority_issue() {
  for (int i = 0; i < 32; ++i) {
    float shm[32];
    for (int j = 0; j < 32; ++j) {
      shm[i] = i;
    }
    for (int j = 0; j < 32; ++j) {
      shm[i * j] += 32;
    }
  }
}
