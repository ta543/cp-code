// time-limit: 4000
// problem-url: https://codeforces.com/contest/1931/problem/G
// 

#include<bits/stdc++.h> 
#define int long long
#define mod 998244353
using namespace std;
int qp(int x,int y)
{
  int ans=1;
  for(int i=1,j=x;i<=y;i*=2,j=(j*j)%mod) if(i&y) ans=(ans*j)%mod;
  return ans;
}
int C(int x,int y)
{
  int ans=1;
  for(int i=1;i<=y;i++) ans=(ans*(x+1-i))%mod,ans=(ans*qp(i,mod-2))%mod;
  return ans;
 } 
int a,b,c,d,t;
signed main()
{
  cin>>t;
  for(int ac=1;ac<=t;ac++)
  {
    cin>>a>>b>>c>>d;
    if(a+b==0&&c*d==0)
    {
      puts("1");
      continue;
    }
    if(a+b==0||abs(a-b)>1)
    {
      puts("0");
      continue;
    }
    if(a-b)
    cout<<(C(c+min(a,b),min(a,b))*C(d+min(a,b),min(a,b)))%mod<<endl;
    else
    cout<<(C(c+min(a,b),min(a,b))*C(d+min(a,b)-1,min(a,b)-1)+C(c+min(a,b)-1,min(a,b)-1)*C(d+min(a,b),min(a,b)))%mod<<endl;
  }
  return 0;
}











