[[okl::kernel("(void)")]]  void simple_function([[okl::restrict("(void)")]] const float *inputArray ,
                             float * outputArray,
                             float value,
                             int size)
{
    [[okl::outer("(void)")]]for(int i = 0; i < size; ++i)   {
        outputArray[i] = inputArray[i] + value;
    }
}
