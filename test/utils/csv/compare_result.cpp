#include <unistd.h>
#include <stdio.h>
#include <string>
#include <cstdlib>
#include <iostream>

using namespace std;

int main() {

    char *mycwd(getenv("PWD"));
    char *pythonIntrepreter="python";
    char *calledPython="/compare_results.py";
    char *pythonArgs[]={pythonIntrepreter,mycwd,calledPython,NULL};
    execvp(pythonIntrepreter,pythonArgs);
    perror("Python execution");

  return 0;
}