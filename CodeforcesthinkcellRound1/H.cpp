// time-limit: 5000
// problem-url: https://codeforces.com/contest/1930/problem/H
// Interactive Mex Tree

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <vector>
using namespace std;

using ll = long long;
using db = long double;
using str = string;

// pairs
using pi = pair<int, int>;
using pl = pair<ll, ll>;
using pd = pair<db, db>;
#define mp make_pair
#define f first
#define s second

#define tcT template <class T
#define tcTU tcT, class U

tcT > using V = vector<T>;
tcT, size_t SZ > using AR = array<T, SZ>;
using vi = V<int>;
using vb = V<bool>;
using vl = V<ll>;
using vd = V<db>;
using vs = V<str>;
using vpi = V<pi>;
using vpl = V<pl>;
using vpd = V<pd>;

// vectors
#define sz(X) int((X).size())
#define bg(X) begin(X)
#define all(X) bg(X), end(X)
#define rall(X) X.rbegin(), X.rend()
#define sor(X) sort(all(X))
#define rsz resize
#define ins insert
#define pb push_back
#define eb emplace_back
#define ft front()
#define bk back()

#define lb lower_bound
#define ub upper_bound
tcT > int lwb(V<T> &a, const T &b) { return int(lb(all(a), b) - bg(a)); }
tcT > int upb(V<T> &a, const T &b) { return int(ub(all(a), b) - bg(a)); }

// loops
#define FOR(i, a, b) for (int i = (a); i < (b); ++i)
#define F0R(i, a) FOR(i, 0, a)
#define ROF(i, a, b) for (int i = (b)-1; i >= (a); --i)
#define R0F(i, a) ROF(i, 0, a)
#define rep(a) F0R(_, a)
#define each(a, X) for (auto &a : X)

const int MOD = 998244353;  // 1e9+7;
const int MX = (int)2e5 + 5;
const ll BIG = 1e18;  // not too close to LLONG_MAX
const db PI = acos((db)-1);
const int dx[4]{1, 0, -1, 0}, dy[4]{0, 1, 0, -1};  // for every grid problem!!
mt19937 rng((uint32_t)chrono::steady_clock::now().time_since_epoch().count());
template <class T> using pqg = priority_queue<T, vector<T>, greater<T>>;

// bitwise ops
constexpr int pct(int X) { return __builtin_popcount(X); }  // # of bits set
constexpr int bits(int X) {

    return X == 0 ? 0 : 31 - __builtin_clz(X);
}
constexpr int p2(int X) { return 1 << X; }
constexpr int msk2(int X) { return p2(X) - 1; }

ll cdiv(ll a, ll b) {
    return a / b + ((a ^ b) > 0 && a % b);
}  // divide a by b rounded up
ll fdiv(ll a, ll b) {
    return a / b - ((a ^ b) < 0 && a % b);
}  // divide a by b rounded down

tcT > bool ckmin(T &a, const T &b) {
    return b < a ? a = b, 1 : 0;
}  // set a = min(a,b)
tcT > bool ckmax(T &a, const T &b) {
    return a < b ? a = b, 1 : 0;
}  // set a = max(a,b)

tcTU > T fstTrue(T lo, T hi, U f) {
    ++hi;
    assert(lo <= hi);  // assuming f is increasing
    while (lo < hi) {  // find first indeX such that f is true
        T mid = lo + (hi - lo) / 2;
        f(mid) ? hi = mid : lo = mid + 1;
    }
    return lo;
}
tcTU > T lstTrue(T lo, T hi, U f) {
    --lo;
    T d = 1;
    while (lo + d < hi) {
        if (!f(lo + d)) {
            hi = lo + d - 1;
            break;
        }
        d *= 2;
    }
    assert(lo <= hi);  // assuming f is decreasing
    while (lo < hi) {  // find first indeX such that f is true
        T mid = lo + (hi - lo + 1) / 2;
        f(mid) ? lo = mid : hi = mid - 1;
    }
    return lo;
}
tcT > void remDup(vector<T> &v) {  // sort and remove duplicates
    sort(all(v));
    v.erase(unique(all(v)), end(v));
}
tcTU > void safeErase(T &t, const U &u) {
    auto it = t.find(u);
    assert(it != end(t));
    t.erase(it);
}

inline namespace IO {
#define SFINAE(x, ...)                                                         \
    template <class, class = void> struct x : std::false_type {};              \
    template <class T> struct x<T, std::void_t<__VA_ARGS__>> : std::true_type {}

SFINAE(DefaultI, decltype(std::cin >> std::declval<T &>()));
SFINAE(DefaultO, decltype(std::cout << std::declval<T &>()));
SFINAE(IsTuple, typename std::tuple_size<T>::type);
SFINAE(Iterable, decltype(std::begin(std::declval<T>())));

template <auto &is> struct Reader {
    template <class T> void Impl(T &t) {
        if constexpr (DefaultI<T>::value) is >> t;
        else if constexpr (Iterable<T>::value) {
            for (auto &X : t) Impl(X);
        } else if constexpr (IsTuple<T>::value) {
            std::apply([this](auto &...args) { (Impl(args), ...); }, t);
        } else static_assert(IsTuple<T>::value, "No matching type for read");
    }
    template <class... Ts> void read(Ts &...ts) { ((Impl(ts)), ...); }
};

template <class... Ts> void re(Ts &...ts) { Reader<cin>{}.read(ts...); }
#define def(t, args...)                                                        \
    t args;                                                                    \
    re(args);

template <auto &os, bool debug, bool print_nd> struct Writer {
    string comma() const { return debug ? "," : ""; }
    template <class T> constexpr char Space(const T &) const {
        return print_nd && (Iterable<T>::value or IsTuple<T>::value) ? '\n'
                                                                     : ' ';
    }
    template <class T> void Impl(T const &t) const {
        if constexpr (DefaultO<T>::value) os << t;
        else if constexpr (Iterable<T>::value) {
            if (debug) os << '{';
            int i = 0;
            for (auto &&X : t)
                ((i++) ? (os << comma() << Space(X), Impl(X)) : Impl(X));
            if (debug) os << '}';
        } else if constexpr (IsTuple<T>::value) {
            if (debug) os << '(';
            std::apply(
                [this](auto const &...args) {
                    int i = 0;
                    (((i++) ? (os << comma() << " ", Impl(args)) : Impl(args)),
                     ...);
                },
                t);
            if (debug) os << ')';
        } else static_assert(IsTuple<T>::value, "No matching type for print");
    }
    template <class T> void ImplWrapper(T const &t) const {
        if (debug) os << "\033[0;31m";
        Impl(t);
        if (debug) os << "\033[0m";
    }
    template <class... Ts> void ps(Ts const &...ts) const {
        ((Impl(ts)), ...);
    }
    template <class F, class... Ts>
    void print_with_sep(const std::string &sep, F const &f,
                        Ts const &...ts) const {
        ImplWrapper(f), ((os << sep, ImplWrapper(ts)), ...), os << '\n';
    }
    void print_with_sep(const std::string &) const { os << '\n'; }
};

template <class... Ts> void pr(Ts const &...ts) {
    Writer<cout, false, true>{}.ps(ts...);
}
template <class... Ts> void ps(Ts const &...ts) {
    Writer<cout, false, true>{}.print_with_sep(" ", ts...);
}
}  // namespace IO

inline namespace Debug {
template <typename... Args> void err(Args... args) {
    Writer<cerr, true, false>{}.print_with_sep(" | ", args...);
}
template <typename... Args> void errn(Args... args) {
    Writer<cerr, true, true>{}.print_with_sep(" | ", args...);
}

void err_prefix(str func, int line, string args) {
    cerr << "\033[0;31m\u001b[1mDEBUG\033[0m"
         << " | "
         << "\u001b[34m" << func << "\033[0m"
         << ":"
         << "\u001b[34m" << line << "\033[0m"
         << " - "
         << "[" << args << "] = ";
}

#ifdef LOCAL
#define dbg(args...) err_prefix(__FUNCTION__, __LINE__, #args), err(args)
#define dbgn(args...) err_prefix(__FUNCTION__, __LINE__, #args), errn(args)
#else
#define dbg(...)
#define dbgn(args...)
#endif

const auto beg_time = std::chrono::high_resolution_clock::now();

double time_elapsed() {
    return chrono::duration<double>(std::chrono::high_resolution_clock::now() -
                                    beg_time)
        .count();
}
}  // namespace Debug

inline namespace FileIO {
void setIn(str s) { freopen(s.c_str(), "r", stdin); }
void setOut(str s) { freopen(s.c_str(), "w", stdout); }
void setIO(str s = "") {
    cin.tie(0)->sync_with_stdio(0);
    cout << fixed << setprecision(12);
    if (sz(s)) setIn(s + ".in"), setOut(s + ".out");
}
}

#define len(X) ll(X.size())
#define elif else if
 
#define eb emplace_back
#define stoi stoll
 
#define MIN(V) *min_element(all(v))
#define MAX(V) *max_element(all(v))
#define LB(C, X) distance((C).begin(), lower_bound(all(C), (X)))
#define UB(C, X) distance((C).begin(), upper_bound(all(C), (X)))
#define UNIQUE(X) \
  sort(all(X)), X.erase(unique(all(X)), X.end()), X.shrink_to_fit()
 
#define INT(...)   \
  int __VA_ARGS__; \
  read(__VA_ARGS__)
#define LL(...)   \
  ll __VA_ARGS__; \
  read(__VA_ARGS__)
#define STR(...)      \
  string __VA_ARGS__; \
  read(__VA_ARGS__)
#define CHAR(...)   \
  char __VA_ARGS__; \
  read(__VA_ARGS__)
#define DBL(...)      \
  double __VA_ARGS__; \
  read(__VA_ARGS__)
 
#define VEC(type, name, size) \
  vector<type> name(size);    \
  read(name)
#define VV(type, name, h, w)                     \
  vector<vector<type>> name(h, vector<type>(w)); \
  read(name)
 
void YES(bool tc = 1) { ps(tc ? "YES" : "NO"); }
void NO(bool tc = 1) { YES(!tc); }
void Yes(bool tc = 1) { ps(tc ? "Yes" : "No"); }
void No(bool tc = 1) { Yes(!tc); }
void yes(bool tc = 1) { ps(tc ? "yes" : "no"); }
void no(bool tc = 1) { yes(!tc); }

// START
 
/**
 * Description: wraps a lambda so it can call itself
 * Source: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0200r0.html
 */
 
namespace std {
 
template <class Fun> class y_combinator_result {
  Fun fun_;
 
  public:
  template <class T>
  explicit y_combinator_result(T &&fun) : fun_(std::forward<T>(fun)) {}
 
  template <class... Args> decltype(auto) operator()(Args &&...args) {
    return fun_(std::ref(*this), std::forward<Args>(args)...);
  }
};
 
template <class Fun> decltype(auto) y_combinator(Fun &&fun) {
  return y_combinator_result<std::decay_t<Fun>>(std::forward<Fun>(fun));
}
 
}  // namespace std
 
void y_comb_demo() {
  cout << y_combinator([](auto gcd, int a, int b) -> int {
    return b == 0 ? a : gcd(b, a % b);
  })(20, 30)
       << "\n";  // outputs 10
}
 
void solve(int tc) {
  def(int, N, Q);
  V<vi> adj(N + 1);
  rep(N - 1) {
    def(int, u, v);
    adj.at(u).pb(v);
    adj.at(v).pb(u);
  }
  vi depth(N + 1), par(N + 1);
  y_combinator([&](auto dfs, int x) -> void {
    for (int y : adj[x]) {
      depth.at(y) = depth.at(x) + 1;
      par.at(y) = x;
      adj.at(y).erase(find(all(adj.at(y)), x));
      dfs(y);
    }
  })(1);
  AR<vi, 2> a, pos;
  F0R(i, 2) { pos.at(i).rsz(N + 1); }
  {
    y_combinator([&](auto dfs_fwd, int x) -> void {
      for (int y : adj[x]) { dfs_fwd(y); }
      pos.at(0).at(x) = sz(a.at(0));
      a.at(0).pb(x);
    })(1);
  }
  {
    y_combinator([&](auto dfs_bwd, int x) -> void {
      R0F(i, sz(adj[x])) {
        int y = adj[x][i];
        dfs_bwd(y);
      }
      pos.at(1).at(x) = sz(a.at(1));
      a.at(1).pb(x);
    })(1);
  }
  dbg(a);
  F0R(i, 2) assert(sz(a.at(i)) == N);
  ps(a);
  cout.flush();
  auto get_path = [&](int u, int l) {
    vi ans;
    while (u != l) {
      assert(depth[u] > depth[l]);
      ans.pb(u);
      u = par[u];
    }
    return ans;
  };
  auto lca = [&](int u, int v) {
    while (u != v) {
      if (depth[u] < depth[v]) swap(u, v);
      u = par[u];
    }
    return u;
  };
 
  rep(Q) {
    def(int, u, v);
    if (pos.at(0).at(u) > pos.at(0).at(v)) swap(u, v);
    int l = lca(u, v);
    vi _path_u = get_path(u, l);
    vi _path_v = get_path(v, l);
    vb in_path(N + 1);
    for (int x : _path_u) in_path.at(x) = 1;
    for (int x : _path_v) in_path.at(x) = 1;
    in_path.at(l) = 1;
    vb queried(N + 1);
    int ans = N;
    auto query = [&](int t, int l, int r) {
      assert(0 <= t && t <= 1);
      if (l > r) return;
      assert(0 <= l && l <= r && r < N);
      ps("?", 1 + t, 1 + l, 1 + r);
      cout.flush();
      FOR(i, l, r + 1) { queried.at(a.at(t).at(i)) = 1; }
      def(int, res);
      ans = min(ans, res);
    };
    auto make_queries = [&](int t, vi p1, vi p2) {
      p2.pb(l);
      if (sz(p1)) {
        assert(pos.at(t).at(p1.bk) < pos.at(t).at(p2.ft));
        query(t, 0, pos.at(t).at(p1.ft) - 1);
        query(t, pos.at(t).at(p1.bk) + 1, pos.at(t).at(p2.ft) - 1);
      } else {
        query(t, 0, pos.at(t).at(p2.ft) - 1);
      }
      if (t == 0) query(t, pos.at(t).at(l) + 1, N - 1);
    };
    make_queries(0, _path_u, _path_v);
    make_queries(1, _path_v, _path_u);
    dbg(in_path, queried);
    FOR(i, 1, N + 1) assert(in_path.at(i) != queried.at(i));
    ps("!", ans);
    cout.flush();
    def(int, trash);
    assert(trash == 1);
  }
}

int main() {
    setIO();
    int TC;
    re(TC);
    FOR(i, 1, TC + 1) solve(i);
}












