#include <stdio.h>
#include <stdlib.h>


#include <iostream>
using namespace std;

int main()
{
    int res = system("/usr/bin/python /home/neel/sree/latest/hipDNN/test/utils/csv/compare_results.py");
    if (res != 0){
        cout << "Exit code was:" << res << "\n";
     }

  return 0;
}
