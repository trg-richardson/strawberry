#include <functional>
#include <queue>
#include <utility>

using cpp_pq = std::priority_queue<std::pair<double, long>,std::vector<std::pair<double, long>>,std::function<bool(std::pair<double,long>,std::pair<double,long>)>>;