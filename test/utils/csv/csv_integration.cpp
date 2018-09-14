#include "csv_integration.hpp"

 void write_to_csv(std::string filename, std::string str, std::string testname,std::uint64_t timer_t){
	std::fstream fs2;
	fs2.open(filename, std::ios::in);
	std::fstream fs;
	fs.open(filename, std::ios_base::app);	
   if (!fs2.is_open()) {
        
        fs << "Test_name"
         << ","
         << "Output" 
         << ","
         <<"Excecution Time (milliseconds)"
         << std::endl;
   }
  fs << testname << ",";
  fs << str <<",";
  fs << timer_t << std::endl;
  fs.close();
}
