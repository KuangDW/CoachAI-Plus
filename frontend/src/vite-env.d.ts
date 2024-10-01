/// <reference types="vite/client" />

declare module "*.csv" {
  export default <{ [key: string]: any }>Array;
}